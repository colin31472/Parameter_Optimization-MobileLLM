import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tiktoken
from transformers import AutoTokenizer
import math
import torch._dynamo

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

data_dir = "data.txt"
text = open(data_dir, 'r', encoding='utf-8').read() # load all the data as simple string

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

# convert our text data into tokenized tensor
data = torch.tensor(
    tokenizer(text, return_tensors="pt", truncation=True)["input_ids"][0],
    dtype=torch.long, 
    device=device
)


#################### Data Loader ####################

train_batch_size = 4  
eval_batch_size = 1  
context_length = 64  # number of tokens processed in a single batch
train_split = 0.8  # percentage of data to use from total data for training

# split data into trian and eval
n_data = len(data)
train_data = data[:int(n_data * train_split)]
eval_data = data[int(n_data * train_split):]

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

d_model = 576 # Embedding dimension
d_ff = 1536 # Hidden dimension in feed-forward network
n_heads = 9 # number of self-attention heads
n_kv_heads = 3 # number of KV heads
n_layers = 30 # number of gpt blocks/layers

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

    def forward(self, logits):
        # 1. Attention
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)

        # 2. Dropout
        logits = self.dropout(adn_logits)

        # 3. Feed Forward with double up-projections and SwiGLU
        gate = self.gate_proj(logits)                    # 첫 번째 up projection
        gate = gate * torch.sigmoid(gate * 1.0)          # SwiGLU activation (beta = 1.0)
        up = self.up_proj(logits)                        # 두 번째 up projection
        up = up * torch.sigmoid(up * 1.0)                # SwiGLU activation (beta = 1.0)
        hidden = gate * up                               # Elementwise multiply
        logits = self.down_proj(hidden)                  # Down projection
        logits = self.ln2(logits + adn_logits)          # Residual + Layer Norm
        
        return logits
    
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, n_layers):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
        self.wpe = PositionalEncoding(context_length, d_model) # word positional encodings
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, n_kv_heads) for _ in  range(n_layers)
        ])
        self.linear1 = nn.Linear(d_model, vocab_size)
        
        ### Embedding Sharing
        self.wte.weight = self.linear1.weight

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

m = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, n_layers=n_layers).to(device)
torch._dynamo.config.suppress_errors = True
m = torch.compile(m)

# Model Parameter 수 계산
print(m)
print(f"Total Parameters: {sum(p.numel() for p in m.parameters() if p.requires_grad) / 1_000_000:.1f}M")


#################### Training ####################

# Hyperparameters
lr = 0.001
epochs = 100
eval_steps = 10
optim = optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

train_loss = {}

for ep in range(epochs):
    xb, yb = train_loader.get_batch()

    logits, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1)
    
    optim.step()
    scheduler.step()
    train_loss[ep] = loss.item()

    if ep % eval_steps == 0 or ep == epochs-1:
        m.eval()
        with torch.no_grad():
            xvb, yvb = eval_loader.get_batch()
            _, e_loss = m(xvb, yvb)

            print(f"Epoch: {ep}\tlr: {lr}\ttrain_loss: {loss:.4f}\teval_loss: {e_loss:.4f}")
        m.train()
        
with torch.no_grad():
    input = torch.tensor(tokenizer.encode("Icis"), dtype=torch.long, device=device).unsqueeze(0)
    print(m.generate(input, max_new_tokens=200)[0])