import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import math
import tiktoken
import gradio as gr
import os

# Model definition (using Lightning structure from v7)
# hyperparameters
BATCH_SIZE = 40
BLOCK_SIZE = 256
MAX_ITERS = int(480000 * 64 / BATCH_SIZE)
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 20
EVAL_ITERS = 5
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
SLIDING_WINDOW_LEN = 128

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
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.o_proj = nn.Linear(head_size, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # batch size, sequence length, embedding dimension (N_EMBD)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)
        value = self.value(x)  # (B, T, head_size)
        output = F.scaled_dot_product_attention(
            q, k, value, attn_mask=None, dropout_p=DROPOUT, is_causal=True
        )
        output = self.o_proj(output)
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [FlashAttentionHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

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
            nn.Dropout(DROPOUT),
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
        x_flat = x.view(B * T, C)  # Flatten tokens to (batch*tokens, d_model)

        # Gating
        gate_scores = self.gate(x_flat)  # (B*T, num_experts)
        topk_scores, topk_indices = torch.topk(
            gate_scores, self.num_experts_per_token, dim=-1
        )  # (B*T, k)
        topk_probs = F.softmax(topk_scores, dim=-1)  # (B*T, k), normalized

        # Output buffer
        out = torch.zeros_like(x_flat)

        # For each expert: route only the tokens assigned to it
        for expert_id, expert in enumerate(self.experts):
            # Find where this expert is selected
            mask = topk_indices == expert_id  # (B*T, k)
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


class GPT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.token_embed_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embed_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)  # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embed_table(idx)  # (B, T, N_EMBD)
        position_emb = self.position_embed_table(torch.arange(T, device=idx.device))

        x = token_emb + position_emb  # (B, T, N_EMBD)
        x = self.blocks(x)  # (B, T, N_EMBD)
        x = self.ln_f(x)  # (B, T, N_EMBD)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
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
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Determine device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)  # use GPU if available

# Try to load the latest v7 model, fallback to v6 if needed
model_paths = [
    # "models/model_v7_lightning_final.pth",  # Final model from v7 training (best choice)
    "models/gpt-v7-epoch=00-step=97000-val_loss=3.99.ckpt",  # Latest checkpoint with best loss
    "models/gpt-v7-epoch=00-step=88000-val_loss=3.98.ckpt",  # Another good checkpoint
    "models/gpt-v7-epoch=00-step=1000-val_loss=3.99.ckpt",  # Other checkpoints
    "models/gpt-v7-epoch=00-step=5000-val_loss=4.00.ckpt",  # Lower number steps, better loss
    "models/gpt-v7-epoch=00-step=4000-val_loss=4.23.ckpt",  # Earlier checkpoint
    "models/gpt-v7-epoch=00-step=3500-val_loss=4.41.ckpt",  # Earlier checkpoint
    "models/gpt-v7-epoch=00-step=1000-val_loss=4.32.ckpt",  # Earlier checkpoint
    "models/gpt-v7-epoch=00-step=1000-val_loss=4.69.ckpt",  # Earlier checkpoint
    "models/gpt-v7-epoch=00-step=500-val_loss=4.51.ckpt",  # Earlier checkpoint
    "models/gpt-v7-epoch=00-step=500-val_loss=4.37.ckpt",  # Earlier checkpoint
    "models/model_v7_lightning.pth",  # Alternative final model from v7
    "models/model_v7_lightning_interrupted.pth",  # Interrupted model from v7 training
    "models/model_v6_flash_attn.pth",  # Fallback to v6
]

# Find the first existing model file with the best available loss
model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path:
    print(f"Loading model from: {model_path}")

    # Check if the model file is a Lightning checkpoint
    if model_path.endswith(".ckpt"):
        # Load as a Lightning checkpoint
        model = GPT.load_from_checkpoint(model_path)
    else:
        # Load as a regular PyTorch state dict
        model = GPT()
        state_dict = torch.load(model_path, map_location=device)

        # Handle case where the model was saved with torch.compile (has _orig_mod. prefix)
        # The state_dict might have been saved with torch.compile which adds _orig_mod. prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                # Remove the _orig_mod. prefix
                new_key = key[len("_orig_mod.") :]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
else:
    print("No model file found. Please train the model first.")
    # Initialize an untrained model for now
    model = GPT().to(device)
    model.eval()

# Compile model for better performance (if available)
try:
    model = torch.compile(model)
    print("Model compiled successfully with torch.compile")
except Exception as e:
    print(f"Could not compile model: {e}")


def generate_text(prompt, max_tokens, temperature, top_k):
    if model_path is None:
        return "No model found. Please train the model first."

    # Encode the prompt
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # Generate text
    with torch.no_grad():
        generated_idx = model.generate(
            idx, max_tokens, temperature=temperature, top_k=top_k
        )

    # Decode the generated text
    generated_text = decode(generated_idx[0].tolist())
    return generated_text[len(prompt) :]  # Return only the generated part


# Create Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            lines=5, label="Input Prompt", placeholder="Enter your text prompt here..."
        ),
        gr.Slider(1, 500, value=100, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
        gr.Slider(1, 100, value=50, label="Top K"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Text Generation with Lightning Transformer Model (v7)",
    description="Generate text using a trained transformer model with Lightning structure. Adjust the parameters to control the output.",
)

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
