import torch
import torch.nn as nn
from torch.nn import functional as F
import math, time, os
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention import flex_attention
import tiktoken
with open('./lecture/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print(f'Length of dataset in characters: {len(text)}')
from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
# This gives you cleaned, plain text articles1
#print(dataset['train'][100]['text'][:500])  # Print the first 500 characters of the first article
#print(dataset['train'][600000])
characters = sorted(list(set(text)))
vocab_size = len(characters)
#print("All the unique characters:", ''.join(characters))
vocab_size = tiktoken.get_encoding("gpt2").n_vocab
#encoder: string to integer
def encode(string):
    return tiktoken.get_encoding("gpt2").encode(string)
    # return [characters.index(c) for c in string]

#decoder: integer to string
def decode(index):
  return tiktoken.get_encoding("gpt2").decode(index)
  # return ''.join([characters[i] for i in index])
string = "hello there"
#print(encode(string))
#print(decode(encode(string)))
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
train_ratio = 0.9
n = int(train_ratio * len(data))
train_data = data[:n]
val_data = data[n:]
#print(train_data.shape, val_data.shape)
# hyperparameters
#batch_size =16# how many independent sequences will we process in parallel?
#block_size = 256# what is the maximum context length for predictions?
#max_iters = int(100000* 64/batch_size) # how many batches to train on
#eval_interval = 500 # how often to evaluate the model
#learning_rate = 3e-4 # learning rate for optimizer
#device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
#eval_iters = 200 # how many batches to use for evaluation
#n_embd = 384 # embedding dimension
#n_head = 6 # number of attention heads
#n_layer = 6 # number of transformer blocks
#dropout = 0.2 # dropout rate
#sliding_window_len =64 

load_previous = True
batch_size =6# how many independent sequences will we process in parallel?
block_size =512# what is the maximum context length for predictions?
max_iters = int(25000* 64/batch_size) # how many batches to train on
eval_interval = 500 # how often to evaluate the model
learning_rate = 3e-4 # learning rate for optimizer
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 200 # how many batches to use for evaluation
n_embd = 512 # embedding dimension
n_head = 8 # number of attention heads
n_layer = 8 #number of transformer blocks
dropout = 0.2 # dropout rate
sliding_window_len =128
# def get_batch(is_train = True):
# def get_batch(is_train = True):
#     data = train_data if is_train else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     # find batch_size random starting indices in the data
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     # get block_size length sequences starting from those indices
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     # for testing, offset start index by 1 to preduct the next character
#     x, y = x.to(device), y.to(device)
#     return x, y

num_dataset_articles = len(dataset['train'])
def get_batch(is_train = True):
    data = dataset['train'] if is_train else dataset['validation']
    ix = torch.randint(num_dataset_articles, (batch_size,)).view(batch_size,)
    # #print(ix)
    # find batch_size random starting indices in the data
    x_list = []
    y_list = []
    for i in ix:
        # #print(f"ix: {i}")
        # #print(data[100])
        i = int(i)
        # #print(data[i])
        # #print(data[i]['text'][:100])
        article = data[i]['text']
        # #print(article[:100])
        article_ids = encode(article)
        # print(len(article_ids))
        while len(article_ids) < block_size + 2:
            article_ids = encode(article+data[torch.randint(num_dataset_articles, (1,)).item()]['text'])
        # #print(f"len article ids: {len(article_ids)-block_size-1}")
        start_idx = torch.randint(0, len(article_ids) - block_size - 1, (1,)).item()
        x_list.append(torch.tensor(article_ids[start_idx:start_idx+block_size], dtype=torch.long))
        y_list.append(torch.tensor(article_ids[start_idx+1:start_idx+block_size+1], dtype=torch.long))
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    x, y = x.to(device), y.to(device)
    return x, y
#print(torch.tril(torch.ones(9,9)) - torch.tril(torch.ones(9,9), -2))
#print(torch.tril(torch.ones(9,9)))
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
class SelfAttentionHead(nn.Module):
  def causal_mask(self, b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

  def __init__(self, head_size):
    self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.o_proj = nn.Linear(head_size, n_embd, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)- torch.tril(torch.ones(block_size, block_size), -sliding_window_len)))
    self.dropout = nn.Dropout(dropout)
    # self.block_mask = create_block_mask(self.causal_mask, 1, 1, block_size,block_size, device=self.device)
  
  def forward(self, x):
    B, T, C = x.shape # batch size, sequence length, embedding dimension (n_embd)
    k = self.key(x)   # (B, T, head_size)
    q = self.query(x)
    
    #compute attention scores
    weights = torch.matmul(q, k.transpose(-2, -1)) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    weights = F.softmax(weights, dim=-1) # (B, T, T)
    weights = self.dropout(weights)
    
    value = self.value(x) # (B, T, head_size)
    # output = flex_attention.flex_attention(q, k, value,block_mask=self.block_mask)
    # output = self.o_proj(output)
    out = torch.matmul(weights, value) # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
    return out
    # return output
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList(SelfAttentionHead(head_size) for _ in range(num_heads))
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
class FFN(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )
    
  def forward(self, x):
    return self.net(x)

    
class MoEFFN(nn.Module):
  def __init__(self, n_embd, num_experts=4, num_experts_per_token=2):
    super().__init__()
    self.num_experts_per_token = num_experts_per_token
    self.num_experts = num_experts
    self.experts = nn.ModuleList([FFN(n_embd) for _ in range(num_experts)])
    self.gate = nn.Linear(n_embd, num_experts)
    
  def forward(self, x):
    B, T, C = x.shape
    gate_scores = F.softmax(self.gate(x), dim=-1) # (B, T, num_experts)
    expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1) # (B, T, C, num_experts)
    # #print(expert_outputs.shape, gate_scores.shape)
    topk_scores, topk_indices = torch.topk(gate_scores, self.num_experts_per_token, dim=-1) # (B, T, 2)
    top_probs = F.softmax(topk_scores, dim=-1) # (B, T, 2)
    expert_outputs = torch.gather(expert_outputs, 3, topk_indices.unsqueeze(2).expand(-1, -1, C, -1)) # (B, T, C, 2)
    out = (expert_outputs * top_probs.unsqueeze(2)).sum(dim=-1)  # (B, T, C)
    return out

class EfficientMoEFFN(nn.Module):
    def __init__(self, n_embd, num_experts=4, num_experts_per_token=2):
        super().__init__()
        self.num_experts_per_token = num_experts_per_token
        self.num_experts = num_experts
        self.experts = nn.ModuleList([FFN(n_embd) for _ in range(num_experts)])
        self.gate = nn.Linear(n_embd, num_experts)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B*T, C)  # Flatten tokens to (batch*tokens, d_model)

        # Gating
        gate_scores = self.gate(x_flat)   # (B*T, num_experts)
        topk_scores, topk_indices = torch.topk(
            gate_scores, self.num_experts_per_token, dim=-1
        )  # (B*T, k)
        topk_probs = F.softmax(topk_scores, dim=-1)  # (B*T, k), normalized

        # Output buffer
        out = torch.zeros_like(x_flat)

        # For each expert: route only the tokens assigned to it
        for expert_id, expert in enumerate(self.experts):
            # Find where this expert is selected
            mask = (topk_indices == expert_id)  # (B*T, k)
            if not mask.any():
                continue # if it's not part of the top k selected experts for any token, skip it

            token_ids, which_slot = mask.nonzero(as_tuple=True)

            # Select actual tokens
            tokens_for_expert = x_flat[token_ids]

            # Apply expert FFN
            expert_out = expert(tokens_for_expert)  # (num_tokens, C)

            # Scale by probability
            probs = topk_probs[token_ids, which_slot].unsqueeze(-1)
            expert_out = expert_out * probs

            # Scatter-add back to output buffer
            out.index_add_(0, token_ids, expert_out)

        return out.view(B, T, C)
class Block(nn.Module):
  # block where you have mha and feedforward then layer normalization
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    # self.ffwd = FeedForward(n_embd)
    self.ffwd =EfficientMoEFFN(n_embd, num_experts=4)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
class LanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embed_table = nn.Embedding(vocab_size, n_embd)
    self.position_embed_table = nn.Embedding(block_size, n_embd)
    self.rotary_emb = RotaryPositionalEmbedding(n_embd)
    self.blocks = nn.Sequential(
      *[Block(n_embd, n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_emb = self.token_embed_table(idx) # (B, T, n_embd)
    position_emb = self.position_embed_table(torch.arange(T, device=device))
    rotary_emb = self.rotary_emb(T, device=device)

    # x = token_emb + position_emb # (B, T, n_embd)
    # print(f"shape rotary emb: {rotary_emb.shape}")
    rotary_emb = rotary_emb.unsqueeze(0).expand(B, -1, -1)  # (1, T, n_embd) -> (B, T, n_embd)
    # print(f"shape rotary emb after unsqueeze and expand: {rotary_emb.shape}")
    # print(f"tokenembed shape: {token_emb.shape}")
    x = token_emb * rotary_emb
    x = self.blocks(x) # (B, T, n_embd)
    x = self.ln_f(x) # (B, T, n_embd)
    logits = self.lm_head(x) # (B, T, vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens, print_characters=False):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # if(print_characters):
      #   print(f"\r{decode(idx[-1].tolist())}", end="", flush=True) 
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      if print_characters:
        print(decode(idx[0].tolist()[-1:]), end="", flush=True)  
    return idx

model = LanguageModel().to(device)
if(load_previous):
  model.load_state_dict(torch.load("./models/model_v5_rotary_embedding.pth"))
  if(model):
    print("loaded state dict")
print(sum(p.numel() for p in model.parameters())/1e6, 'Million Model Parameters')

optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def generate_streaming(model, context, max_new_tokens):
    max_new_tokens = max_new_tokens-1
    for _ in range(max_new_tokens):
      context = model.generate(context, max_new_tokens=1)
      generated = decode(context[0].tolist())[-1]
      print(f"{generated}", end="", flush=True)
    print()
    return generated

for iter in range(max_iters):
  if(iter % eval_interval ==0):
    torch.save(model.state_dict(), './models/checkpoints/model_v5_rotary_embedding.pth')
  x_data, y_data = get_batch(is_train=True)
  logits, loss = model(x_data, y_data)
  optim.zero_grad(set_to_none=True)
  loss.backward()
  optim.step()
  print(f"\r Iter {iter+1}/{max_iters}, Loss: {loss.item()}", end="", flush=True)
#train loop
torch.save(model.state_dict(), './models/model_v5_rotary_embedding.pth')
# model.load_state_dict(torch.load('./models/model_lecture_style.pth'))
# model.load_state_dict(torch.load('./models/model_v5_rotary_embedding.pth'))
# context = torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=500, print_characters=True)[0].tolist()))
# text_to_be_continued = "ROMEO"
# context = torch.tensor(encode(text_to_be_continued), dtype=torch.long, device=device).unsqueeze(0)
# print(decode(model.generate(context, max_new_tokens=5000, print_characters=True)[0].tolist()))