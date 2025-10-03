import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import gradio as gr
import os

# Model definition (copied from your training script)
# hyperparameters
batch_size = 24  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = int(160000 * 64 / batch_size)  # how many batches to train on
eval_interval = 500  # how often to evaluate the model
learning_rate = 3e-4  # learning rate for optimizer
device = 'mps' if torch.backends.mps.is_available(
) else 'cuda' if torch.cuda.is_available() else 'cpu'  # use GPU if available
eval_iters = 200  # how many batches to use for evaluation
n_embd = 384  # embedding dimension
n_head = 6  # number of attention heads
n_layer = 6  # number of transformer blocks
dropout = 0.2  # dropout rate
sliding_window_len = 128

# Get vocab size from tiktoken
vocab_size = tiktoken.get_encoding("gpt2").n_vocab

# Encoder/decoder functions


def encode(string):
    return tiktoken.get_encoding("gpt2").encode(string)


def decode(index):
    return tiktoken.get_encoding("gpt2").decode(index)


class FlashAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.o_proj = nn.Linear(head_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch size, sequence length, embedding dimension (n_embd)
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)
        value = self.value(x)  # (B, T, head_size)
        output = F.scaled_dot_product_attention(
            q, k, value, attn_mask=None, dropout_p=dropout, is_causal=True)
        output = self.o_proj(output)
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(FlashAttentionHead(head_size)
                                   for _ in range(num_heads))
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
                continue  # if it's not part of the top k selected experts for any token, skip it

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
        self.ffwd = EfficientMoEFFN(n_embd, num_experts=4)
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
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embed_table(idx)  # (B, T, n_embd)
        position_emb = self.position_embed_table(
            torch.arange(T, device=idx.device))

        x = token_emb + position_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Load the model
model = LanguageModel().to(device)
model_path = "models/model_v6_flash_attn.pth"

# Check if model file exists
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
else:
    print(f"Model file not found at {
          model_path}. Please train the model first.")

# Compile model for better performance
model = torch.compile(model)


def generate_text(prompt, max_tokens, temperature, top_k):
    if not os.path.exists(model_path):
        return "Model not found. Please train the model first."

    # Encode the prompt
    idx = torch.tensor(encode(prompt), dtype=torch.long,
                       device=device).unsqueeze(0)

    # Generate text
    with torch.no_grad():
        generated_idx = model.generate(
            idx, max_tokens, temperature=temperature, top_k=top_k)

    # Decode the generated text
    generated_text = decode(generated_idx[0].tolist())
    return generated_text[len(prompt):]  # Return only the generated part


# Create Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=5, label="Input Prompt",
                   placeholder="Enter your text prompt here..."),
        gr.Slider(1, 500, value=100, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
        gr.Slider(1, 100, value=50, label="Top K")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Text Generation with Transformer Model",
    description="Generate text using a trained transformer model. Adjust the parameters to control the output."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
