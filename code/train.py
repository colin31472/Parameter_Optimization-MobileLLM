# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tiktoken
from transformers import AutoTokenizer
import math
import torch._dynamo
from datasets import load_dataset
import os
os.environ['TORCH_INDUCTOR_DISABLE_TRITON'] = '1'
import numpy as np

import matplotlib.pyplot as plt

# import sys
# sys.stdout.reconfigure(line_buffering=True)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    # print(f"GPU: {torch.cuda.get_device_name()}")
    # print(f"CUDA 버전: {torch.version.cuda}")
    torch.cuda.empty_cache()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#################### Tokenizer ####################

train_batch_size = 4  
eval_batch_size = 1  
context_length = 128  # number of tokens processed in a single batch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

dataset = load_dataset("wikitext", "wikitext-103-v1")

# 비어있지 않은 텍스트만 필터링
def filter_non_empty_text(example):
    return len(example["text"].strip()) > 0

# 필터링된 데이터셋 생성
dataset = dataset.filter(filter_non_empty_text)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    # 빈 텍스트 필터링
    filtered_text = [text for text in examples["text"] if text.strip()]
    
    # 각 텍스트를 개별적으로 토큰화
    tokenized_texts = tokenizer(
        filtered_text, 
        truncation=True, 
        max_length=context_length, 
        padding='max_length', 
        return_tensors="pt"
    )
    
    return {"input_ids": tokenized_texts["input_ids"]}

tokenized_train = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_valid = dataset["validation"].map(tokenize_function, batched=True, remove_columns=dataset["validation"].column_names)
tokenized_test = dataset["test"].map(tokenize_function, batched=True, remove_columns=dataset["test"].column_names)

train_data = torch.tensor(tokenized_train["input_ids"]).to(device)
eval_data = torch.tensor(tokenized_valid["input_ids"]).to(device)
test_data = torch.tensor(tokenized_test["input_ids"]).to(device)

train_data = train_data.view(-1).to(device)
eval_data = eval_data.view(-1).to(device)
test_data = test_data.view(-1).to(device)

# 토크나이저 출력 결과 확인
sample_text = dataset["train"][0]["text"]
# 샘플 텍스트 출력
print("샘플 텍스트:", sample_text)
print("텍스트 길이:", len(sample_text))
print("텍스트 비었는지 여부:", not bool(sample_text.strip()))
# tokens = tokenizer(sample_text, truncation=True, max_length=128, return_tensors="pt")


#################### Data Loader ####################

class DataLoader:
    def __init__(self, tokens, batch_size, context_length, device) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length
        
        # 데이터가 한 배치보다 작으면 반복해서 늘림
        if len(self.tokens) < b * c:
            repeats = (b * c // len(self.tokens)) + 1
            self.tokens = self.tokens.repeat(repeats)
        
        # 현재 위치가 데이터 길이를 초과하면 처음으로 돌아감
        if self.current_position + b * c >= len(self.tokens):
            self.current_position = 0
        
        start_pos = self.current_position
        end_pos = start_pos + b * c + 1
        
        d = self.tokens[start_pos:end_pos]
        
        # 배치 크기가 부족하면 처음부터 추가
        if len(d) < b * c + 1:
            needed = b * c + 1 - len(d)
            d = torch.cat([d, self.tokens[:needed]])
            
        x = (d[:-1]).view(b, c).to(self.device)
        y = (d[1:]).view(b, c).to(self.device)

        self.current_position = (self.current_position + b * c) % len(self.tokens)
        return x, y

train_loader = DataLoader(train_data, train_batch_size, context_length, device)
eval_loader = DataLoader(eval_data, eval_batch_size, context_length, device)
test_loader = DataLoader(test_data, eval_batch_size, context_length, device)


#################### Model ####################

d_model = 576 # Embedding dimension
d_ff = 1536 # Hidden dimension in feed-forward network
n_heads = 9 # number of self-attention heads
n_kv_heads = 3 # number of KV heads
n_layers = 30 # number of gpt blocks/layers
layer_sharing = True

num_thinking_steps = 3  # number of thinking steps
memory_gate = 0.01     # memory gate for KV memory update


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads  # 각 KV 헤드가 몇 개의 Q 헤드를 담당하는지
        self.head_dim = d_model // n_heads
        
        assert (n_heads * self.head_dim == d_model)
        assert (n_heads % n_kv_heads == 0)  # n_heads는 n_kv_heads로 나누어떨어져야 함
        
        # Q는 원래 헤드 수만큼, K,V는 줄어든 헤드 수만큼 생성
        self.query = nn.Linear(d_model, d_model)  # (d_model -> d_model)
        self.key = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.value = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        
        # output을 위한 Layer
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape
        
        # Q, K, V 생성 및 헤드 분할
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # KV 헤드 반복하여 Q 헤드 수만큼 확장
        K = repeat_kv(K, self.n_kv_groups)  # (B, n_heads, seq_length, head_dim)
        V = repeat_kv(V, self.n_kv_groups)  # (B, n_heads, seq_length, head_dim)
        
        # 이후는 기존과 동일한 attention 계산
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Masking (upper triangular matrix)
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask = mask.to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        # attention weights * V
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # 차원 순서 되돌린 뒤 헤드 합치기
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)
        
        out = self.fc_out(attention_output)
        
        return out

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV 헤드를 반복하여 Q 헤드 수만큼 확장하는 함수"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model):
        super().__init__()
        pe = torch.zeros(1, context_length, d_model)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        # Feed Forward Network with double up-projections
        self.gate_proj = nn.Linear(d_model, d_ff)  # 첫 번째 up projection
        self.up_proj = nn.Linear(d_model, d_ff)    # 두 번째 up projection
        self.down_proj = nn.Linear(d_ff, d_model)  # down projection
        
        self.att = MultiHeadAttention(d_model, n_heads, n_kv_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        
        # Separate trainable alpha parameters for each residual connection
        self.alpha1 = nn.Parameter(torch.tensor(0.0))  # attention residual용
        self.alpha2 = nn.Parameter(torch.tensor(0.0))  # feedforward residual용
        
        # Sigmoid func to keep alphas between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits):
        # 1. Attention
        att_logits = self.att(logits)
        
        # First residual connection using alpha1
        alpha1 = self.sigmoid(self.alpha1)
        residual = alpha1 * logits + (1 - alpha1) * att_logits
        adn_logits = self.ln1(residual)

        # 2. Dropout
        logits = self.dropout(adn_logits)

        # 3. Feed Forward with double up-projections and SwiGLU
        gate = self.gate_proj(logits)                    
        gate = gate * torch.sigmoid(gate * 1.0)         
        up = self.up_proj(logits)                        
        up = up * torch.sigmoid(up * 1.0)               
        hidden = gate * up                              
        ff_logits = self.down_proj(hidden)                  
        
        # Second residual connection using alpha2
        alpha2 = self.sigmoid(self.alpha2)
        residual = alpha2 * adn_logits + (1 - alpha2) * ff_logits
        logits = self.ln2(residual)
        
        return logits
    
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, n_layers, layer_sharing):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
        self.wpe = PositionalEncoding(context_length, d_model) # word positional encodings
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, n_kv_heads) for _ in  range(n_layers)
        ])
        self.linear1 = nn.Linear(d_model, vocab_size)
        
        ### Embedding Sharing
        self.wte.weight = self.linear1.weight
        
        ### Layer Sharing
        self.layer_sharing = layer_sharing
        
        self.gradient_checkpointing = False
        
    def gradient_checkpointing_enable(self):
        """Gradient checkpointing 활성화"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Gradient checkpointing 비활성화"""
        self.gradient_checkpointing = False
        
    def forward(self, inputs, targets=None):        
        # 1. Embedding
        logits = self.wte(inputs) # Token embedding
        logits = self.wpe(logits) # Position embedding
        
        # 2. N GPT Blocks with Layer Sharing
        for block in self.blocks:
            # Immediate Block-wise Sharing
            logits = block(logits)
            if self.layer_sharing:
                logits = block(logits)
        
        # 3. final result
        logits = self.linear1(logits)
        
        # 4. Compute Loss (if training)
        loss = None
        if targets is not None:
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            loss = F.cross_entropy(logits_view, targets_view)
        
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        output = inputs.clone()
        
        # Limit Context Length
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)
            if current_seq_length > context_length:
                inputs = inputs[:, -context_length:]
            
            logits, _ = self(inputs)  
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=1)            
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
            
        return [tokenizer.decode(out.tolist(), skip_special_tokens=True) for out in output]

class GPTWithDeepThinking(GPT):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, n_layers, layer_sharing,
                 num_thinking_steps=5, memory_gate=0.01):
        super().__init__(vocab_size, d_model, n_heads, n_kv_heads, n_layers, layer_sharing)
        
        self.num_thinking_steps = num_thinking_steps
        self.memory_gate = memory_gate
        self.kv_memories = None
    
    def reset_memories(self):
        """Reset KV Memories"""
        self.kv_memories = None
    
    def _get_block_memories(self, batch_size, seq_length):
        """Initialize KV Memories"""
        return {
            'key': torch.zeros(batch_size, self.n_heads, seq_length, 
                             self.d_model // self.n_heads).to(next(self.parameters()).device),
            'value': torch.zeros(batch_size, self.n_heads, seq_length,
                               self.d_model // self.n_heads).to(next(self.parameters()).device)
        }
    
    def _update_memories(self, current_k, current_v, block_idx):
        """KV Memory Update"""
        if self.kv_memories is None:
            self.kv_memories = [{
                'key': current_k,
                'value': current_v
            } for _ in range(len(self.blocks))]
        else:
            old_k = self.kv_memories[block_idx]['key']
            old_v = self.kv_memories[block_idx]['value']
            
            new_k = self.memory_gate * current_k + (1 - self.memory_gate) * old_k
            new_v = self.memory_gate * current_v + (1 - self.memory_gate) * old_v
            
            self.kv_memories[block_idx] = {
                'key': new_k,
                'value': new_v
            }
    
    def _forward_thinking(self, x):
        """Deep Thinking Stage"""
        B, seq_length = x.shape
        
        # 1. Embedding
        hidden_states = self.wte(x)
        hidden_states = self.wpe(hidden_states)
        
        # 2. Multiple thinking iterations
        for step in range(self.num_thinking_steps):
            current_states = hidden_states
            
            for idx, block in enumerate(self.blocks):
                att = block.att
                
                # Input states for residual
                residual_pre_att = current_states
                
                # Generate current Q, K, V
                Q = att.query(current_states).view(B, seq_length, att.n_heads, -1).permute(0, 2, 1, 3)
                K = att.key(current_states).view(B, seq_length, att.n_kv_heads, -1).permute(0, 2, 1, 3)
                V = att.value(current_states).view(B, seq_length, att.n_kv_heads, -1).permute(0, 2, 1, 3)
                
                # Expand K, V for GQA
                K = repeat_kv(K, att.n_kv_groups)
                V = repeat_kv(V, att.n_kv_groups)
                
                if step > 0:
                    # Concatenate with previous memories
                    memory = self.kv_memories[idx]
                    K_combined = torch.cat([memory['key'], K], dim=2)
                    V_combined = torch.cat([memory['value'], V], dim=2)
                    
                    attention_scores = torch.matmul(Q, K_combined.transpose(-2, -1)) / math.sqrt(att.head_dim)
                    
                    combined_length = K_combined.size(2)
                    mask = torch.triu(torch.ones(seq_length, combined_length), diagonal=1).bool()
                    mask = mask.to(attention_scores.device)
                    attention_scores = attention_scores.masked_fill(mask, float('-inf'))
                    
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    attention_output = torch.matmul(attention_weights, V_combined)
                else:
                    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(att.head_dim)
                    
                    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
                    mask = mask.to(attention_scores.device)
                    attention_scores = attention_scores.masked_fill(mask, float('-inf'))
                    
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    attention_output = torch.matmul(attention_weights, V)
                
                # Update memories
                self._update_memories(K, V, idx)
                
                # Process attention output
                attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
                attention_output = attention_output.view(B, seq_length, -1)
                attention_output = att.fc_out(attention_output)
                
                # First enhanced residual connection with layer norm
                alpha1 = block.sigmoid(block.alpha1)
                current_states = alpha1 * residual_pre_att + (1 - alpha1) * attention_output
                current_states = block.ln1(current_states)
                current_states = block.dropout(current_states)
                
                # Save states for second residual
                residual_pre_ffn = current_states
                
                # FFN with SwiGLU
                gate = block.gate_proj(current_states)
                gate = gate * torch.sigmoid(gate * 1.0)
                up = block.up_proj(current_states)
                up = up * torch.sigmoid(up * 1.0)
                hidden = gate * up
                ff_output = block.down_proj(hidden)
                
                # Second enhanced residual connection with layer norm
                alpha2 = block.sigmoid(block.alpha2)
                current_states = alpha2 * residual_pre_ffn + (1 - alpha2) * ff_output
                current_states = block.ln2(current_states)
                
                # Layer sharing if enabled
                if self.layer_sharing:
                    # Repeat the same block
                    residual_pre_att = current_states
                    
                    # Same process for shared layer
                    Q = att.query(current_states).view(B, seq_length, att.n_heads, -1).permute(0, 2, 1, 3)
                    K = att.key(current_states).view(B, seq_length, att.n_kv_heads, -1).permute(0, 2, 1, 3)
                    V = att.value(current_states).view(B, seq_length, att.n_kv_heads, -1).permute(0, 2, 1, 3)
                    
                    K = repeat_kv(K, att.n_kv_groups)
                    V = repeat_kv(V, att.n_kv_groups)
                    
                    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(att.head_dim)
                    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
                    mask = mask.to(attention_scores.device)
                    attention_scores = attention_scores.masked_fill(mask, float('-inf'))
                    
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    attention_output = torch.matmul(attention_weights, V)
                    
                    attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
                    attention_output = attention_output.view(B, seq_length, -1)
                    attention_output = att.fc_out(attention_output)
                    
                    # Enhanced residual connections for shared layer
                    alpha1 = block.sigmoid(block.alpha1)
                    current_states = alpha1 * residual_pre_att + (1 - alpha1) * attention_output
                    current_states = block.ln1(current_states)
                    current_states = block.dropout(current_states)
                    
                    residual_pre_ffn = current_states
                    
                    gate = block.gate_proj(current_states)
                    gate = gate * torch.sigmoid(gate * 1.0)
                    up = block.up_proj(current_states)
                    up = up * torch.sigmoid(up * 1.0)
                    hidden = gate * up
                    ff_output = block.down_proj(hidden)
                    
                    alpha2 = block.sigmoid(block.alpha2)
                    current_states = alpha2 * residual_pre_ffn + (1 - alpha2) * ff_output
                    current_states = block.ln2(current_states)
            
            hidden_states = current_states
        
        return hidden_states
    
    def _forward_inference(self, x):
        """Test Stage"""
        B, seq_length = x.shape
        
        # 1. Embedding
        hidden_states = self.wte(x)
        hidden_states = self.wpe(hidden_states)
        
        # 2. Use accumulated memories for inference
        for idx, block in enumerate(self.blocks):
            att = block.att
            memory = self.kv_memories[idx]
            
            # Input states for residual
            residual_pre_att = hidden_states
            
            # Generate only Q, use stored K, V
            Q = att.query(hidden_states).view(B, seq_length, att.n_heads, -1).permute(0, 2, 1, 3)
            K = memory['key']
            V = memory['value']
            
            # Calculate attention using stored memories
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(att.head_dim)
            
            mask = torch.triu(torch.ones(seq_length, K.size(2)), diagonal=1).bool()
            mask = mask.to(attention_scores.device)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
            
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V)
            
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            attention_output = attention_output.view(B, seq_length, -1)
            attention_output = att.fc_out(attention_output)
            
            # First enhanced residual connection
            alpha1 = block.sigmoid(block.alpha1)
            hidden_states = alpha1 * residual_pre_att + (1 - alpha1) * attention_output
            hidden_states = block.ln1(hidden_states)
            hidden_states = block.dropout(hidden_states)
            
            # Save states for second residual
            residual_pre_ffn = hidden_states
            
            # FFN
            gate = block.gate_proj(hidden_states)
            gate = gate * torch.sigmoid(gate * 1.0)
            up = block.up_proj(hidden_states)
            up = up * torch.sigmoid(up * 1.0)
            hidden = gate * up
            ff_output = block.down_proj(hidden)
            
            # Second enhanced residual connection
            alpha2 = block.sigmoid(block.alpha2)
            hidden_states = alpha2 * residual_pre_ffn + (1 - alpha2) * ff_output
            hidden_states = block.ln2(hidden_states)
            
            # Layer sharing
            if self.layer_sharing:
                residual_pre_att = hidden_states
                
                Q = att.query(hidden_states).view(B, seq_length, att.n_heads, -1).permute(0, 2, 1, 3)
                attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(att.head_dim)
                attention_scores = attention_scores.masked_fill(mask, float('-inf'))
                
                attention_weights = torch.softmax(attention_scores, dim=-1)
                attention_output = torch.matmul(attention_weights, V)
                
                attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
                attention_output = attention_output.view(B, seq_length, -1)
                attention_output = att.fc_out(attention_output)
                
                alpha1 = block.sigmoid(block.alpha1)
                hidden_states = alpha1 * residual_pre_att + (1 - alpha1) * attention_output
                hidden_states = block.ln1(hidden_states)
                hidden_states = block.dropout(hidden_states)
                
                residual_pre_ffn = hidden_states
                
                gate = block.gate_proj(hidden_states)
                gate = gate * torch.sigmoid(gate * 1.0)
                up = block.up_proj(hidden_states)
                up = up * torch.sigmoid(up * 1.0)
                hidden = gate * up
                ff_output = block.down_proj(hidden)
                
                alpha2 = block.sigmoid(block.alpha2)
                hidden_states = alpha2 * residual_pre_ffn + (1 - alpha2) * ff_output
                hidden_states = block.ln2(hidden_states)
        
        return hidden_states
    
    def forward(self, x, targets=None, is_thinking=False):
        if is_thinking:
            hidden_states = self._forward_thinking(x)
        else:
            if self.kv_memories is None:
                raise RuntimeError("Must run thinking stage before inference.")
            hidden_states = self._forward_inference(x)
        
        logits = self.linear1(hidden_states)
        
        loss = None
        if targets is not None:
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            loss = F.cross_entropy(logits_view, targets_view)
        
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        output = inputs.clone()
        
        # Deep Thinking stage for initial context
        _, _ = self(inputs, is_thinking=True)
        
        # Limit Context Length
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)
            if current_seq_length > context_length:
                inputs = inputs[:, -context_length:]
                # Reset memories when context is truncated
                self.reset_memories()
                _, _ = self(inputs, is_thinking=True)
            
            # Inference stage for token generation
            logits, _ = self(inputs, is_thinking=False)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
        
        # Reset memories after generation
        self.reset_memories()
        
        return [tokenizer.decode(out.tolist(), skip_special_tokens=True) for out in output]

m = GPTWithDeepThinking(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    n_layers=n_layers,
    layer_sharing=layer_sharing,
    num_thinking_steps=num_thinking_steps,
    memory_gate=memory_gate
).to(device)

torch._dynamo.config.suppress_errors = True
# m = torch.compile(m)

# 모델 파라미터 수 계산
print(m)
print(f"Total Parameters: {sum(p.numel() for p in m.parameters() if p.requires_grad) / 1_000_000:.1f}M")


#################### Training ####################

def train_model(model, optimizer, scheduler, train_loader, eval_loader, 
                start_epoch, epochs, eval_steps=10, best_eval_loss=float('inf')):
    train_loss = {}
    
    # Enable gradient checkpointing & Mixed precision 준비
    model.gradient_checkpointing_enable()
    scaler = torch.amp.GradScaler()
    
    # 데이터 크기 출력 및 steps_per_epoch 계산
    n_tokens = len(train_loader.tokens)
    batch_tokens = train_loader.batch_size * train_loader.context_length
    steps_per_epoch = max(1, n_tokens // batch_tokens)
    
    print(f"Total tokens: {n_tokens}")
    print(f"Tokens per batch: {batch_tokens}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    global_step = 0
    
    for ep in range(start_epoch, epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for step in range(steps_per_epoch):
            # GPU 캐시 정리
            if step % 10 == 0:  # 매 10 스텝마다
                torch.cuda.empty_cache()
            
            xb, yb = train_loader.get_batch()
            
            # Reset memories for new batch
            model.reset_memories()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda'):
                # Deep Thinking stage
                _, _ = model(xb, yb, is_thinking=True)
                
                # Test stage with loss calculation
                logits, loss = model(xb, yb, is_thinking=False)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Scale loss and call backward
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            # Step optimizer and scaler
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 일정 스텝마다 평가 수행
            if global_step % eval_steps == 0:
                avg_step_loss = epoch_loss / num_batches
                
                # avg loss가 nan인지 검사
                if math.isnan(avg_step_loss):
                    print(f"NaN average loss detected at epoch {ep}, step {step}. Terminating training...")
                    return model, optimizer, train_loss, best_eval_loss
                
                model.eval()
                with torch.no_grad():
                    eval_ppl = evaluate_wikitext(model, eval_loader)
                    
                    # eva; perplexity가 nan인지 검사
                    if math.isnan(eval_ppl):
                        print(f"NaN evaluation perplexity detected at epoch {ep}, step {step}. Terminating training...")
                        return model, optimizer, train_loss, best_eval_loss
                    
                    print(f"Epoch: {ep}, Step: {global_step}/{steps_per_epoch}\t"
                          f"lr: {scheduler.get_last_lr()[0]:.6f}\t"
                          f"train_loss: {avg_step_loss:.4f}\t"
                          f"eval_ppl: {eval_ppl:.4f}")
                    
                    save_model(model, optimizer, global_step, avg_step_loss, best_eval_loss, save_dir='checkpoints')
                    if eval_ppl < best_eval_loss:
                        best_eval_loss = eval_ppl
                        
                
                model.train()
        
        avg_loss = epoch_loss / num_batches
        train_loss[ep] = avg_loss
        
    return model, optimizer, train_loss, best_eval_loss
        

#################### Save & Load ####################

def save_model(model, optimizer, step, loss, best_eval_loss, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for filename in os.listdir(save_dir):
        if filename.startswith('Deep_'):
            os.remove(os.path.join(save_dir, filename))

    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_eval_loss': best_eval_loss,
    }
    
    path = os.path.join(save_dir, f'Deep_Thinking_model_step_{step}.pt')
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    return model, optimizer, step, loss, best_eval_loss


#################### Evaluation ####################

def evaluate_wikitext(model, eval_loader, num_batches=100):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            x, y = eval_loader.get_batch()
            model.reset_memories()
            _, _ = model(x, y, is_thinking=True)  # Deep Thinking stage
            _, loss = model(x, y, is_thinking=False)  # Test stage
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

## Evaluation - ARC
#### Deep Thinking ouput으로 수정 필요

def prepare_arc_dataset(split='validation'):
    """
    ARC 데이터셋을 준비하는 함수
    """
    # ARC-Easy와 ARC-Challenge 데이터셋 로드
    arc_easy = load_dataset("ai2_arc", "ARC-Easy", split=split)
    arc_challenge = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    
    return arc_easy, arc_challenge

def process_arc_example(example, tokenizer, max_length=128):
    question = example['question']
    choices = example['choices']['text']
    answer_key = example.get('answerKey', None)

    # 정답 인덱스 추출
    if answer_key is not None:
        answer_idx = ord(answer_key) - ord('A')
    else:
        raise ValueError("Missing answerKey in example")

    inputs = []
    for choice in choices:
        text = f"Question: {question} Answer: {choice}"
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        inputs.append(encoded['input_ids'].squeeze(0))  # Remove batch dimension

    return inputs, answer_idx

def evaluate_arc(model, tokenizer, dataset, batch_size=8):
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for example in dataset:
            inputs, answer_idx = process_arc_example(example, tokenizer)  # 기존 함수 사용
            scores = []
            
            # Reset memories for new example
            model.reset_memories()
            
            # Deep Thinking stage 
            for input_tensor in inputs:
                input_tensor = input_tensor.to(device)
                input_tensor = input_tensor.unsqueeze(0)
                _, _ = model(input_tensor, is_thinking=True)
            
            # Test stage
            for input_tensor in inputs:
                input_tensor = input_tensor.to(device)
                input_tensor = input_tensor.unsqueeze(0)
                outputs, _ = model(input_tensor, is_thinking=False)
                
                # 기존 scoring 로직 유지
                decoded_input = tokenizer.decode(input_tensor[0], skip_special_tokens=False)
                answer_start = decoded_input.find("Answer:") + len("Answer:")
                
                if answer_start == -1:
                    raise ValueError("'Answer:' not found in the input sequence.")
                
                answer_start_token = tokenizer.encode(decoded_input[:answer_start], add_special_tokens=False)
                answer_start_idx = len(answer_start_token)
                answer_end_idx = (input_tensor != tokenizer.eos_token_id).sum(dim=1)[0]
                
                logits = outputs[0, answer_start_idx:answer_end_idx]
                next_tokens = input_tensor[0, answer_start_idx+1:answer_end_idx+1]
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_scores = torch.gather(log_probs, 1, next_tokens.unsqueeze(1)).squeeze(1)
                score = token_scores.mean().item()
                
                scores.append(score)
            
            predicted_idx = np.argmax(scores)
            if predicted_idx == answer_idx:
                correct += 1
            total += 1
            
    accuracy = correct / total
    return accuracy

def run_evaluation(model, tokenizer):
    """
    ARC-Easy와 ARC-Challenge에 대한 전체 평가 실행
    """
    # 데이터셋 준비
    arc_easy, arc_challenge = prepare_arc_dataset()
    
    # ARC-Easy 평가
    easy_accuracy = evaluate_arc(model, tokenizer, arc_easy)
    print(f"ARC-Easy Accuracy: {easy_accuracy:.2%}")
    
    # ARC-Challenge 평가
    challenge_accuracy = evaluate_arc(model, tokenizer, arc_challenge)
    print(f"ARC-Challenge Accuracy: {challenge_accuracy:.2%}")
    
    return {
        'arc_easy_accuracy': easy_accuracy,
        'arc_challenge_accuracy': challenge_accuracy
    }


def plot_alphas_from_checkpoint(checkpoint_path, save_dir='plot_alpha'):
    """
    체크포인트에서 alpha 값을 추출하여 시각화하고 저장
    """
    # 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    step = checkpoint['step']
    
    # alpha 값 추출
    alpha1_values = []
    alpha2_values = []
    
    for i in range(n_layers):
        alpha1_key = f'blocks.{i}.alpha1'
        alpha2_key = f'blocks.{i}.alpha2'
        
        alpha1 = torch.sigmoid(state_dict[alpha1_key]).item()
        alpha2 = torch.sigmoid(state_dict[alpha2_key]).item()
        
        alpha1_values.append(alpha1)
        alpha2_values.append(alpha2)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    layers = list(range(1, n_layers + 1))
    
    plt.plot(layers, alpha1_values, 'b-o', label='Alpha1 (Attention)', linewidth=2, markersize=6)
    plt.plot(layers, alpha2_values, 'r-o', label='Alpha2 (FFN)', linewidth=2, markersize=6)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Alpha Value', fontsize=12)
    plt.title(f'Alpha Values Across Layers', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # x축 레이블 설정
    plt.xticks(layers[::2])  # 2 레이어마다 표시
    
    # y축 범위 설정
    plt.ylim([0, 1])
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(save_dir, f'alpha_values_step_{step}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Alpha values plot saved to {save_path}")


#################### Main ####################

# Training Configuration
load_from_checkpoint = True
checkpoint_path = 'checkpoints/Deep_Thinking_model_step_79500.pt'

# Hyperparameters
lr = 2e-3
epochs = 1
eval_steps = 500  # 500 스텝마다 평가 수행

model = m
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

start_epoch = 0
best_eval_loss = float('inf')

if load_from_checkpoint:
    try:
        model, optimizer, start_epoch, loss, best_eval_loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming from step {start_epoch} with best eval loss: {best_eval_loss}")
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        load_from_checkpoint = False

# 남은 에폭 수에 맞춰 scheduler 설정
remaining_epochs = epochs - start_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

# Training
# model, optimizer, train_loss, best_eval_loss = train_model(
#     model, optimizer, scheduler, train_loader, eval_loader,
#     start_epoch=start_epoch, epochs=epochs, 
#     eval_steps=eval_steps, best_eval_loss=best_eval_loss
# )

# wikitext_ppl = evaluate_wikitext(model, eval_loader)
test_perplexity = evaluate_wikitext(model, test_loader, num_batches=100)
print(f"WikiText-2 test Perplexity: {test_perplexity:.2f}")

plot_alphas_from_checkpoint(checkpoint_path)


# #################### ARC Evaluation Main Code ####################

# checkpoint_path = 'checkpoints/Deep_Thinking_model_step_0.pt'

# # 모델과 옵티마이저 초기화
# model = m
# model = model.to(device)
# optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.1)
    
# # 체크포인트에서 모델 로드
# try:
#     model, optimizer, epoch, loss, best_eval_loss = load_checkpoint(
#         model, optimizer, checkpoint_path
#     )
#     print(f"Successfully loaded model from checkpoint:")
#     print(f"Epoch: {epoch}, Loss: {loss:.4f}, Best Eval Loss: {best_eval_loss:.4f}")
# except FileNotFoundError:
#     print(f"Error: Checkpoint not found at {checkpoint_path}")
#     exit()
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")
#     exit()
    
# model.eval()
    
# # ARC 평가 실행
# print("\nStarting ARC evaluation...")
# try:
#     results = run_evaluation(model, tokenizer)
        
#     # 결과 출력
#     print("\nEvaluation Results:")
#     print("=" * 50)
#     print(f"Checkpoint Epoch: {epoch}")
#     print(f"ARC-Easy Accuracy: {results['arc_easy_accuracy']:.2%}")
#     print(f"ARC-Challenge Accuracy: {results['arc_challenge_accuracy']:.2%}")
#     print("=" * 50)
    
#     os.makedirs('results', exist_ok=True)
        
#     # 결과 저장
#     save_path = 'results/arc_evaluation_results.txt'
#     with open(save_path, 'w') as f:
#         f.write(f"Model checkpoint: {checkpoint_path}\n")
#         f.write(f"Checkpoint Epoch: {epoch}\n")
#         f.write(f"Final Loss: {loss:.4f}\n")
#         f.write(f"Best Eval Loss: {best_eval_loss:.4f}\n")
#         f.write(f"ARC-Easy Accuracy: {results['arc_easy_accuracy']:.2%}\n")
#         f.write(f"ARC-Challenge Accuracy: {results['arc_challenge_accuracy']:.2%}\n")
#     print(f"\nResults saved to {save_path}")
        
# except Exception as e:
#     print(f"Error during evaluation: {e}")