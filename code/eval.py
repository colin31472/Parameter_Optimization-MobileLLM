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
import sys
sys.stdout.reconfigure(line_buffering=True)

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

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

dataset = load_dataset("wikitext", "wikitext-2-v1")

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

train_data = torch.tensor(tokenized_train["input_ids"]).to(device)
eval_data = torch.tensor(tokenized_valid["input_ids"]).to(device)

train_data = train_data.view(-1).to(device)
eval_data = eval_data.view(-1).to(device)

# 토크나이저 출력 결과 확인
sample_text = dataset["train"][0]["text"]
# 샘플 텍스트 출력
print("샘플 텍스트:", sample_text)
print("텍스트 길이:", len(sample_text))
print("텍스트 비어있나?:", not bool(sample_text.strip()))
tokens = tokenizer(sample_text, truncation=True, max_length=128, return_tensors="pt")

#################### Data Loader ####################

train_batch_size = 4  
eval_batch_size = 1  
context_length = 128  # number of tokens processed in a single batch

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


#################### Model ####################

d_model = 768 # Embedding dimension
d_ff = 1536 # Hidden dimension in feed-forward network
n_heads = 12 # number of self-attention heads
n_layers = 12 # number of gpt blocks/layers

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # d_model이 n_heads로 나누어떨어지는지 확인
        assert (n_heads * self.head_dim == d_model)
        
        # Q, K, V
        self.query = nn.Linear(d_model, d_model)  # (d_model -> d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # output을 위한 Layer
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape  # (batch_size, seq_len, d_model)
        
        # Q, K, V 생성 및 헤드 분할
        # 1. Linear 변환
        # 2. View로 헤드 분할
        # 3. Permute로 차원 순서 변경
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 최종 shape: (B, n_heads, seq_length, head_dim)
        
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, n_heads, seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.head_dim)  # Scale factor

        # Masking (upper triangular ma)
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask = mask.to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Attention weights using SoftMax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Attention weight * V
        attention_output = torch.matmul(attention_weights, V)  # (B, n_heads, seq_length, head_dim)
        
        # 1. 차원 순서 되돌리기
        attention_output = attention_output.permute(0, 2, 1, 3)  # (B, seq_length, n_heads, head_dim)

        # 2. 연속적인 메모리 배치를 위한 contiguous
        attention_output = attention_output.contiguous()

        # 3. 헤드 합치기
        attention_output = attention_output.view(B, seq_length, d_model)

        # 4. 최종 output
        out = self.fc_out(attention_output)  # (B, seq_length, d_model)
        
        return out


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
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.att = MultiHeadAttention(d_model, n_heads)

        # Layer Normalization layers
        self.ln1 = nn.LayerNorm(d_model)  # 첫 번째 residual connection 후의 normalization
        self.ln2 = nn.LayerNorm(d_model)  # 두 번째 residual connection 후의 normalization
        self.dropout = nn.Dropout(0.2)     # 20% 확률로 뉴런을 비활성화

    def forward(self, logits):
        # 1. Attention
        att_logits = self.att(logits)      # MultiHead-attention
        adn_logits = self.ln1(logits + att_logits)  # Residual + Layer Norm

        # 2. Dropout
        logits = self.dropout(adn_logits)

        # 3. Feed Forward
        logits = self.fcn(logits)          # Feed-forward network
        logits = self.ln2(logits + adn_logits)  # Residual + Layer Norm

        return logits
    
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
        self.wpe = PositionalEncoding(context_length, d_model) # word positional encodings
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads) for _ in  range(n_layers)
        ])
        self.linear1 = nn.Linear(d_model, vocab_size)
        
        # ### Embedding Sharing
        # self.wte.weight = self.linear1.weight
        
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
        
        # 2. N GPT Blocks
        for block in self.blocks:
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

m = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
torch._dynamo.config.suppress_errors = True
# m = torch.compile(m)

# Model Parameter 수 계산
print(m)
print(f"Total Parameters: {sum(p.numel() for p in m.parameters() if p.requires_grad) / 1_000_000:.1f}M")


#################### Training ####################

def train_model(model, optimizer, scheduler, train_loader, eval_loader, 
                start_epoch, epochs, eval_steps=10, best_eval_loss=float('inf')):
    train_loss = {}
    
    # Enable gradient checkpointing & Mixed precision 준비
    model.gradient_checkpointing_enable()
    scaler = torch.cuda.amp.GradScaler()
    
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
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logits, loss = model(xb, yb)
            
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
                model.eval()
                with torch.no_grad():
                    eval_ppl = evaluate_wikitext(model, eval_loader)
                    print(f"Epoch: {ep}, Step: {global_step}/{steps_per_epoch}\t"
                          f"lr: {scheduler.get_last_lr()[0]:.6f}\t"
                          f"train_loss: {avg_step_loss:.4f}\t"
                          f"eval_ppl: {eval_ppl:.4f}")
                    
                    if eval_ppl < best_eval_loss:
                        best_eval_loss = eval_ppl
                        save_model(model, optimizer, ep, avg_step_loss, best_eval_loss, save_dir='checkpoints')
                
                model.train()
        
        avg_loss = epoch_loss / num_batches
        train_loss[ep] = avg_loss
    
    return model, optimizer, train_loss, best_eval_loss


#################### Save & Load ####################

def save_model(model, optimizer, epoch, loss, best_eval_loss, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for filename in os.listdir(save_dir):
        if filename.endswith('.pt'):
            os.remove(os.path.join(save_dir, filename))

    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_eval_loss': best_eval_loss,
    }
    
    path = os.path.join(save_dir, f'Deep_Thinking_model_epoch_{epoch}.pt')
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    return model, optimizer, epoch, loss, best_eval_loss


#################### Evaluation ####################

def evaluate_wikitext(model, eval_loader, num_batches=100):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            x, y = eval_loader.get_batch()
            _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


#################### Main ####################

# Checkpoint configuration
checkpoint_path = 'checkpoints/Baseline_epoch_0.pt'

# Initialize model and optimizer
model = m  # 모델 아키텍처 
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

# Load checkpoint
try:
    model, optimizer, epoch, loss, best_eval_loss = load_checkpoint(model, optimizer, checkpoint_path)
    print(f"Successfully loaded model from epoch {epoch}")
    print(f"Previous loss: {loss:.4f}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
except FileNotFoundError:
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    exit()

# Evaluate perplexity
model.eval()
wikitext_ppl = evaluate_wikitext(model, eval_loader)
print(f"WikiText-2 Perplexity: {wikitext_ppl:.2f}")