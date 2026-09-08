#!/usr/bin/env python3

from pathlib import Path

import json
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (Dataset, DataLoader,)

sys.path.append(
    str(Path(__file__).parent)
)
from smiles_tokenizer import (load_vocab, encode_smiles)
from smiles_vae import (SmilesVAE)


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path("data/generation/processed/moses_100k_split.csv")
VOCAB_FILE = Path("data/generation/processed/smiles_vocab.json")
MODEL_DIR = Path("models/vae")
REPORT_DIR = Path("reports/generation")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Architecture
# ------------------------------------------------------------

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 128
NUM_LAYERS = 1


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 30
RANDOM_STATE = 42
GRADIENT_CLIP = 5.0

# ------------------------------------------------------------
# KL annealing
#
# Start with almost pure reconstruction.
# Gradually introduce latent-space regularization.
# ------------------------------------------------------------

KL_WARMUP_EPOCHS = 10
MAX_BETA = 0.1


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 70)
print("SMILES VAE TRAINING")
print("=" * 70)
print(f"Device: {device}")


if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# Load data / vocabulary
# ============================================================

df = pd.read_csv(DATA_FILE)
vocab = load_vocab(VOCAB_FILE)
VOCAB_SIZE = len(vocab["token_to_id"])
PAD_ID = vocab["special_token_ids"]["PAD"]
MAX_LENGTH = vocab["max_sequence_length"]

train_df = (df[df["split"] == "train"].reset_index(drop=True))
validation_df = (df[df["split"] == "validation"].reset_index(drop=True))

print()
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Maximum length:  {MAX_LENGTH}")

print(f"Training: {len(train_df)}")
print(f"Validation: {len(validation_df)}")

# ============================================================
# Dataset
# ============================================================
class SmilesDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.smiles = (dataframe["canonical_smiles"].tolist())
        self.vocab = vocab
        self.max_length = vocab["max_sequence_length"]
        self.pad_id = vocab["special_token_ids"]["PAD"]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        ids = encode_smiles(smiles, self.vocab)
        length = len(ids)
        padded = (ids + [self.pad_id] * (self.max_length - length))
        return (torch.tensor(padded, dtype=torch.long), torch.tensor(length, dtype=torch.long))

# ============================================================
# DataLoaders
# ============================================================

train_dataset = SmilesDataset(train_df, vocab)
validation_dataset = SmilesDataset(validation_df, vocab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# Model
# ============================================================

model = SmilesVAE(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=NUM_LAYERS)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ============================================================
# Reconstruction loss
# PAD tokens are ignored.
# ============================================================

reconstruction_criterion = (nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum"))

# ============================================================
# KL loss
# ============================================================

def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl

# ============================================================
# Beta schedule
# ============================================================
def get_beta(epoch):
    if epoch >= KL_WARMUP_EPOCHS:
        return MAX_BETA

    fraction = (epoch / KL_WARMUP_EPOCHS)
    return (MAX_BETA * fraction)

# ============================================================
# One epoch
# ============================================================

def run_epoch(loader, training, beta):
    if training:
        model.train()
    else:
        model.eval()

    total_recon = 0.0
    total_kl = 0.0
    total_tokens = 0
    total_molecules = 0

    for (input_ids, lengths) in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)

        # ---------------------------------------------
        # Teacher forcing
        # Full:
        # <BOS> C C O <EOS> PAD
        # decoder input:
        # <BOS> C C O <EOS>
        # target:
        # C C O <EOS> PAD
        # ---------------------------------------------

        decoder_input = (input_ids[:, :-1])
        decoder_target = (input_ids[:, 1:])

        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            (logits, mu, logvar, _) = model(input_ids, lengths, decoder_input)

            # logits:
            # B x T x V
            reconstruction_loss = (reconstruction_criterion(logits.reshape(-1, VOCAB_SIZE,), decoder_target.reshape(-1),))
            kl_loss = kl_divergence(mu, logvar)

            # Normalize reconstruction loss
            # by number of real target tokens

            non_pad_tokens = (decoder_target != PAD_ID).sum()
            recon_per_token = (reconstruction_loss / non_pad_tokens)
            kl_per_molecule = (kl_loss / input_ids.size(0))
            loss = (recon_per_token + beta * kl_per_molecule)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP,) 
                optimizer.step()

        total_recon += (reconstruction_loss.item())
        total_kl += (kl_loss.item())
        total_tokens += (non_pad_tokens.item())
        total_molecules += (input_ids.size(0))

    mean_recon = (total_recon / total_tokens)
    mean_kl = (total_kl / total_molecules)
    total_loss = (mean_recon + beta * mean_kl)

    return {"loss": total_loss, "reconstruction": mean_recon, "kl": mean_kl}

# ============================================================
# Training loop
# ============================================================

history = []
best_validation_loss = np.inf

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    beta = get_beta(epoch)
    train_metrics = run_epoch(train_loader, training=True, beta=beta)
    validation_metrics = run_epoch(validation_loader, training=False, beta=beta)
    elapsed = (time.time() - start_time)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"beta={beta:.4f} | "
        f"train={train_metrics['loss']:.4f} | "
        f"val={validation_metrics['loss']:.4f} | "
        f"recon={validation_metrics['reconstruction']:.4f} | "
        f"KL={validation_metrics['kl']:.4f} | "
        f"{elapsed:.1f}s"
    )

    history.append(
        {"epoch": epoch, "beta": beta, "train_loss": train_metrics["loss"], 
         "train_reconstruction": train_metrics["reconstruction"], "train_kl": train_metrics["kl"],
         "validation_loss": validation_metrics["loss"], "validation_reconstruction": validation_metrics["reconstruction"],
         "validation_kl": validation_metrics["kl"], "seconds": elapsed,})

    # ========================================================
    # Best checkpoint
    # ========================================================

    if (validation_metrics["loss"] < best_validation_loss):
        best_validation_loss = (validation_metrics["loss"])
        checkpoint = {"model_state_dict": model.state_dict(), "vocab_size": VOCAB_SIZE, "pad_id": PAD_ID,
                      "embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM, "latent_dim": LATENT_DIM,
                      "num_layers": NUM_LAYERS, "epoch": epoch, "validation_loss": best_validation_loss}

        torch.save(checkpoint, MODEL_DIR / "best_smiles_vae.pt")

# ============================================================
# Save final model
# ============================================================

torch.save({"model_state_dict": model.state_dict(), "vocab_size": VOCAB_SIZE, "pad_id": PAD_ID, 
            "embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM, "num_layers": NUM_LAYERS,}, 
            MODEL_DIR / "final_smiles_vae.pt")


# ============================================================
# Save history
# ============================================================

history_df = pd.DataFrame(history)
history_df.to_csv(REPORT_DIR / "vae_training_history.csv", index=False)

# ============================================================
# Save configuration
# ============================================================

config = {"embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM, "latent_dim": LATENT_DIM, "num_layers": NUM_LAYERS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
    "epochs":EPOCHS, "kl_warmup_epochs":KL_WARMUP_EPOCHS, "max_beta":MAX_BETA, "gradient_clip":GRADIENT_CLIP, "random_state":RANDOM_STATE}

with open(REPORT_DIR / "vae_training_config.json", "w") as f:
    json.dump(config, f, indent=4)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_validation_loss:.4f}"
)

print()
print("Saved:")
print(
    "  models/vae/best_smiles_vae.pt"
)

print(
    "  models/vae/final_smiles_vae.pt"
)

print(
    "  reports/generation/"
    "vae_training_history.csv"
)