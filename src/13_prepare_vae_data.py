#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (Dataset, DataLoader)

# Allow imports from src/
sys.path.append(
    str(Path(__file__).parent)
)

from smiles_tokenizer import (load_vocab, encode_smiles, decode_ids)


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path("data/generation/processed/moses_100k.csv")
VOCAB_FILE = Path("data/generation/processed/smiles_vocab.json")
OUTPUT_FILE = Path("data/generation/processed/moses_100k_split.csv")
REPORT_FILE = Path("reports/generation/vae_data_summary.json")

RANDOM_STATE = 42
TRAIN_FRACTION = 0.90
VALIDATION_FRACTION = 0.05
HOLDOUT_FRACTION = 0.05
BATCH_SIZE = 128

# ============================================================
# Load
# ============================================================
df = pd.read_csv(DATA_FILE)
vocab = load_vocab(VOCAB_FILE)

max_sequence_length = vocab["max_sequence_length"]
pad_id = vocab["special_token_ids"]["PAD"]

print("=" * 70)
print("VAE DATA PREPARATION")
print("=" * 70)
print(f"Molecules:           {len(df)}")
print(f"Vocabulary size:     {len(vocab['token_to_id'])}")
print(f"Max sequence length: {max_sequence_length}")

# ============================================================
# Deterministic split
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)
indices = np.arange(len(df))
rng.shuffle(indices)
n_total = len(df)
n_train = int(n_total * TRAIN_FRACTION)
n_validation = int(n_total * VALIDATION_FRACTION)
train_indices = indices[:n_train]
validation_indices = indices[n_train:n_train + n_validation]
holdout_indices = indices[n_train + n_validation:]

df["split"] = None
df.loc[train_indices, "split"] = "train"
df.loc[validation_indices, "split"] = "validation"
df.loc[holdout_indices, "split"] = "holdout"

print()
print("Split:")
print(df["split"].value_counts())

# ============================================================
# Dataset
# ============================================================

class SmilesDataset(Dataset):
    def __init__(self,dataframe,vocab):

        self.smiles = (
            dataframe[
                "canonical_smiles"
            ]
            .tolist()
        )

        self.vocab = vocab

        self.max_length = vocab[
            "max_sequence_length"
        ]

        self.pad_id = vocab[
            "special_token_ids"
        ]["PAD"]


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        ids = encode_smiles(smiles, self.vocab)
        sequence_length = len(ids)
        
        if (sequence_length > self.max_length):
            raise RuntimeError(
                f"Sequence too long: "
                f"{sequence_length}"
            )

        # ---------------------------------------------
        # Pad sequence
        # ---------------------------------------------

        padded_ids = (ids + [self.pad_id] * (self.max_length - sequence_length))
        input_ids = torch.tensor(padded_ids, dtype=torch.long)

        return {"input_ids": input_ids, "length": sequence_length, "smiles": smiles}

# ============================================================
# Build datasets
# ============================================================

train_df = df[df["split"] == "train"].reset_index(drop=True)
validation_df = df[df["split"] == "validation"].reset_index(drop=True)
holdout_df = df[df["split"] == "holdout"].reset_index(drop=True)

train_dataset = SmilesDataset(train_df, vocab)
validation_dataset = SmilesDataset(validation_df, vocab)
holdout_dataset = SmilesDataset(holdout_df, vocab)

# ============================================================
# Test DataLoader
# ============================================================

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
batch = next(iter(train_loader))

print()
print("=" * 70)
print("DATALOADER TEST")
print("=" * 70)


print("input_ids shape:",batch["input_ids"].shape)
print("length shape:",batch["length"].shape,)

print()
print("First sequence length:",batch["length"][0].item())


# ============================================================
# Decode first example
#
# This tests:
#
# SMILES
#   -> tokens
#   -> IDs
#   -> SMILES
#
# and verifies nothing was lost.
# ============================================================

original_smiles = (batch["smiles"][0])
encoded = (batch["input_ids"][0])
decoded_smiles = decode_ids(encoded,vocab)

print()
print("=" * 70)
print("ENCODE / DECODE TEST")
print("=" * 70)

print("Original:",original_smiles,)

print("Decoded: ",decoded_smiles)


if original_smiles != decoded_smiles:
    raise RuntimeError("Encode/decode mismatch.")

print()
print("Encode/decode test PASSED.")


# ============================================================
# Demonstrate decoder input / target
#
# Full sequence:
#
# <BOS> C C O <EOS> PAD PAD
#
# Decoder input:
#
# <BOS> C C O
#
# Target:
#
# C C O <EOS>
#
# During training we shift by one token.
# ============================================================

example = batch["input_ids"][0]
decoder_input = example[:-1]
decoder_target = example[1:]

print()
print("=" * 70)
print("AUTOREGRESSIVE SHIFT")
print("=" * 70)

print("Full tensor length:", len(example))
print("Decoder input length:", len(decoder_input))
print("Decoder target length:", len(decoder_target))

# ============================================================
# Save split manifest
# ============================================================

df.to_csv(OUTPUT_FILE,index=False,)
summary = {"n_total": len(df), "n_train": len(train_df), "n_validation": len(validation_df), "n_holdout": len(holdout_df),
    "vocab_size": len(vocab["token_to_id"]),
    "max_sequence_length": max_sequence_length,
    "batch_size": BATCH_SIZE,
    "random_state": RANDOM_STATE}


REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_FILE, "w") as f:
    json.dump(summary, f, indent=4,)

print()
print("=" * 70)
print("VAE DATA PREPARATION COMPLETE")
print("=" * 70)

print()
print("Saved split manifest: ", OUTPUT_FILE)
print("Saved summary: ", REPORT_FILE)