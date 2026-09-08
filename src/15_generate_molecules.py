#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
sys.path.append(str(Path(__file__).parent))

from smiles_tokenizer import load_vocab
from smiles_vae import SmilesVAE

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")

# ============================================================
# Configuration
# ============================================================

VOCAB_FILE = Path("data/generation/processed/smiles_vocab.json")
DATA_FILE = Path("data/generation/processed/moses_100k_split.csv")
MODEL_FILE = Path("models/vae/best_smiles_vae.pt")
OUTPUT_DIR = Path("data/generation/processed")
REPORT_DIR = Path("reports/generation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_GENERATE = 10_000
BATCH_SIZE = 512

# Lower = safer/more conservative
# Higher = more diverse/random
TEMPERATURE = 1.0
RANDOM_STATE = 42


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("SMILES VAE GENERATION")
print("=" * 70)
print(f"Device:      {device}")
print(f"Generate:    {N_GENERATE}")
print(f"Temperature: {TEMPERATURE}")


# ============================================================
# Vocabulary
# ============================================================

vocab = load_vocab(VOCAB_FILE)
token_to_id = vocab["token_to_id"]
id_to_token = vocab["id_to_token"]

PAD_ID = vocab["special_token_ids"]["PAD"]
BOS_ID = vocab["special_token_ids"]["BOS"]
EOS_ID = vocab["special_token_ids"]["EOS"]
UNK_ID = vocab["special_token_ids"]["UNK"]
MAX_LENGTH = vocab["max_sequence_length"]

# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(MODEL_FILE, map_location=device)
model = SmilesVAE(
    vocab_size=checkpoint["vocab_size"],
    pad_id=checkpoint["pad_id"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    latent_dim=checkpoint["latent_dim"],
    num_layers=checkpoint["num_layers"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
LATENT_DIM = checkpoint["latent_dim"]

print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Latent dimension: {LATENT_DIM}")

# ============================================================
# Convert generated IDs -> SMILES
# ============================================================

def ids_to_smiles(ids):
    tokens = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id in {PAD_ID, BOS_ID, UNK_ID}:
            continue
        tokens.append(id_to_token[str(token_id)])

    return "".join(tokens)

# ============================================================
# Autoregressive generation
# ============================================================

@torch.no_grad()
def generate_batch(batch_size):
    # ---------------------------------------------
    # Sample latent vectors from prior
    # z ~ N(0, I)
    # ---------------------------------------------
    z = torch.randn(batch_size, LATENT_DIM, device=device)

    # Initial decoder hidden state
    hidden = model.latent_to_hidden(z)
    hidden = hidden.view(batch_size, model.num_layers, model.hidden_dim)
    hidden = hidden.permute(1, 0, 2).contiguous()

    # Start every molecule with <BOS>
    current_token = torch.full((batch_size,), BOS_ID, dtype=torch.long, device=device)
    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Maximum generated content length.
    # BOS is not included here.
    for _ in range(MAX_LENGTH - 1):
        embedded = model.embedding(current_token)
        decoder_input = torch.cat([embedded, z], dim=-1)
        decoder_input = decoder_input.unsqueeze(1)
        output, hidden = model.decoder_gru(decoder_input, hidden)
        logits = model.output_layer(output[:, 0, :])
        logits = logits / TEMPERATURE
        # These should never be generated as molecular tokens
        logits[:, PAD_ID] = -float("inf")
        logits[:, BOS_ID] = -float("inf")
        logits[:, UNK_ID] = -float("inf")

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        
        for i in range(batch_size):
            if finished[i]:
                continue

            token_id = next_token[i].item()
            if token_id == EOS_ID:
                finished[i] = True
            else:
                generated[i].append(token_id)
        current_token = next_token
        if finished.all():
            break
    return [ids_to_smiles(ids) for ids in generated]


# ============================================================
# Generate molecules
# ============================================================

generated_smiles = []
print()
print("Generating molecules...")

while len(generated_smiles) < N_GENERATE:
    remaining = N_GENERATE - len(generated_smiles)
    current_batch = min(BATCH_SIZE, remaining)
    generated_smiles.extend(generate_batch(current_batch))
    if len(generated_smiles) % 2000 == 0:
        print(f"  generated {len(generated_smiles)}/{N_GENERATE}")

# ============================================================
# Load TRAINING molecules for novelty calculation
# ============================================================

source_df = pd.read_csv(DATA_FILE)
training_smiles = set(source_df.loc[source_df["split"] == "train", "canonical_smiles"])

# ============================================================
# RDKit validation
# ============================================================

records = []
for smiles in generated_smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        records.append({"generated_smiles": smiles, "valid": False, "canonical_smiles": None})
        continue

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    records.append({"generated_smiles": smiles, "valid": True, "canonical_smiles": canonical})

results = pd.DataFrame(records)

# ============================================================
# Validity
# ============================================================

n_total = len(results)
n_valid = int(results["valid"].sum())
validity = n_valid / n_total

# ============================================================
# Uniqueness
# ============================================================

valid_df = results[results["valid"]].copy()
n_unique = valid_df["canonical_smiles"].nunique()
uniqueness = (n_unique / n_valid if n_valid > 0 else 0.0)

# ============================================================
# Novelty
# ============================================================

unique_valid = set(valid_df["canonical_smiles"].dropna())
novel_smiles = unique_valid - training_smiles
n_novel = len(novel_smiles)
novelty = (n_novel / n_unique if n_unique > 0 else 0.0)
valid_df["novel_vs_training"] = (~valid_df["canonical_smiles"].isin(training_smiles))

# ============================================================
# Save
# ============================================================

results.to_csv(OUTPUT_DIR / "vae_generated_all.csv",index=False)

valid_df.to_csv(OUTPUT_DIR / "vae_generated_valid.csv", index=False)

novel_df = (valid_df[valid_df["novel_vs_training"]].drop_duplicates("canonical_smiles"))
novel_df.to_csv(OUTPUT_DIR / "vae_generated_novel.csv", index=False)


summary = pd.DataFrame([{
    "n_generated": n_total,
    "n_valid": n_valid,
    "validity": validity,
    "n_unique_valid": n_unique,
    "uniqueness": uniqueness,
    "n_novel": n_novel,
    "novelty": novelty,
    "temperature": TEMPERATURE,
}])

summary.to_csv(REPORT_DIR / "vae_generation_summary.csv", index=False)


# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print("GENERATION RESULTS")
print("=" * 70)

print(f"Generated:       {n_total}")
print(f"Valid:           {n_valid}")
print(f"Validity:        {validity:.4f}")

print()
print(f"Unique valid:    {n_unique}")
print(f"Uniqueness:      {uniqueness:.4f}")

print()
print(f"Novel:           {n_novel}")
print(f"Novelty:         {novelty:.4f}")

print()
print("Example valid molecules:")

for smiles in valid_df["canonical_smiles"].drop_duplicates().head(20):
    print(" ", smiles)

print()
print("Saved:")
print("  data/generation/processed/vae_generated_all.csv")
print("  data/generation/processed/vae_generated_valid.csv")
print("  data/generation/processed/vae_generated_novel.csv")
print("  reports/generation/vae_generation_summary.csv")