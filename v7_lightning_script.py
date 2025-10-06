import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset
import math
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Boolean hyperparameters
LOAD_PREVIOUS = True  # Load previous model checkpoint
SAVE_ON_INTERRUPT = True  # Save model when training is interrupted

# Hyperparameters as constants
BATCH_SIZE = 40
BLOCK_SIZE = 256
MAX_ITERS = int(160000 * 64 / BATCH_SIZE)
LEARNING_RATE = 3e-4
EVAL_INTERVAL = BATCH_SIZE/2
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
SLIDING_WINDOW_LEN = 128


class OpenWebTextDataset(Dataset):
    def __init__(self, split="train", max_articles=None):
        self.dataset = load_dataset("Bingsu/openwebtext_20p")
        self.split = split
        self.data = self.dataset[split]
        self.num_articles = len(self.data)
        if max_articles:
            self.num_articles = min(self.num_articles, max_articles)

        # Set up encoding/decoding
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoder.n_vocab

    def __len__(self):
        # Return a large virtual length to allow for many iterations
        return 10**9

    def __getitem__(self, idx):
        # Randomly sample an article
        article_idx = torch.randint(0, self.num_articles, (1,)).item()
        article = self.data[article_idx]["text"]

        # Encode the article
        article_ids = self.encoder.encode(article)

        # If the article is too short, concatenate with more articles
        while len(article_ids) < BLOCK_SIZE + 2:
            additional_idx = torch.randint(0, self.num_articles, (1,)).item()
            additional_article = self.data[additional_idx]["text"]
            article_ids.extend(self.encoder.encode(additional_article))

        # Randomly select a sequence of block_size + 1 tokens
        start_idx = torch.randint(
            0, len(article_ids) - BLOCK_SIZE - 1, (1,)).item()
        sequence = article_ids[start_idx: start_idx + BLOCK_SIZE + 1]

        # Split into input and target
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)

        return x, y


class FlashAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.o_proj = nn.Linear(head_size, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
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
            [FlashAttentionHead(head_size) for _ in range(num_heads)])
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
    def __init__(self, vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.token_embed_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embed_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)  # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embed_table(idx)  # (B, T, N_EMBD)
        position_emb = self.position_embed_table(
            torch.arange(T, device=idx.device))

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('val_loss', loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def generate(self, idx, max_new_tokens, print_characters=False):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            if print_characters:
                encoder = tiktoken.get_encoding("gpt2")
                print(encoder.decode(idx[0].tolist()[-1:]), end="", flush=True)
        return idx


class OpenWebTextDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = OpenWebTextDataset(split="train")
        # Using train split for validation since OpenWebText doesn't have validation
        self.val_dataset = OpenWebTextDataset(split="train")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )


def main():
    # Setup
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab

    # Initialize data module
    data_module = OpenWebTextDataModule(batch_size=BATCH_SIZE)

    # Check if a previous model exists to load based on hyperparameter
    checkpoint_path = './models/model_v6_flash_attn.pth'
    if LOAD_PREVIOUS and os.path.exists(checkpoint_path):
        print("Loading previous checkpoint...")
        # Load the model state dict manually to initialize the model
        model = GPT(vocab_size=vocab_size)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Use strict=False to handle potential mismatches
        model.load_state_dict(state_dict, strict=False)
        print("Previous checkpoint loaded successfully")
    else:
        print("Starting with a new model...")
        model = GPT(vocab_size=vocab_size)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='models/',
        filename='gpt-v7-{epoch:02d}-{step}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        every_n_train_steps=500,  # Save checkpoint every 500 steps
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pl.Trainer(
        max_steps=MAX_ITERS,
        val_check_interval=EVAL_INTERVAL,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed',  # Use mixed precision for faster training
        accumulate_grad_batches=1,  # Gradient accumulation if needed
        gradient_clip_val=1.0,  # Gradient clipping for stability
        devices='auto',  # Use auto to detect available devices (GPU, TPU, CPU)
        accelerator='auto',  # Auto-detect best accelerator
        strategy='auto',  # Auto-detect best strategy
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,  # Set to True for reproducibility but may impact performance
        # resume_from_checkpoint=None,  # Can be set to a checkpoint path to resume training
    )

    # Enable memory optimizations
    torch.set_float32_matmul_precision("high")

    # Compile the model for further optimization (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled successfully with torch.compile")
    except Exception as e:
        print(f"Could not compile model: {e}")

    # Start training with interrupt handling
    try:
        trainer.fit(model, datamodule=data_module)

        # Save final model after training
        final_model_path = './models/model_v7_lightning.pth'
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        print(f"\nModel state saved after training: {final_model_path}")

    except KeyboardInterrupt:
        if SAVE_ON_INTERRUPT:
            print("\nTraining interrupted by user. Saving model...")
            final_model_path = './models/model_v7_lightning_interrupted.pth'
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save(model.state_dict(), final_model_path)
            print(f"\nModel state saved: {final_model_path}")
        else:
            print("\nTraining interrupted by user. Model not saved.")

    finally:
        if SAVE_ON_INTERRUPT:
            # Make sure final checkpoint is saved
            final_model_path = './models/model_v7_lightning_final.pth'
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save(model.state_dict(), final_model_path)
            print(f"\nModel state saved in finally block: {final_model_path}")


if __name__ == "__main__":
    main()
