#!/usr/bin/env python3

from pathlib import Path
import json
import re
from collections import Counter
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "data/generation/processed/moses_100k.csv"
)

OUTPUT_DIR = Path(
    "data/generation/processed"
)

REPORT_DIR = Path(
    "reports/generation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Special tokens
# ============================================================

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
]


# ============================================================
# SMILES tokenizer
#
# Important cases:
#
# Cl
# Br
# [NH+]
# [C@@H]
# %10
#
# are treated as single tokens.
# ============================================================

TOKEN_PATTERN = re.compile(
    r"("
    r"\[[^\[\]]+\]"
    r"|Br"
    r"|Cl"
    r"|Si"
    r"|Se"
    r"|@@?"
    r"|%\d{2}"
    r"|."
    r")"
)


def tokenize_smiles(smiles):

    tokens = TOKEN_PATTERN.findall(
        smiles
    )

    # Important validation:
    # joining tokens should exactly reconstruct
    # the original SMILES.
    reconstructed = "".join(tokens)

    if reconstructed != smiles:
        raise ValueError(
            f"Tokenizer failed:\n"
            f"SMILES:        {smiles}\n"
            f"Reconstructed: {reconstructed}"
        )

    return tokens


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("SMILES TOKENIZER / VOCABULARY")
print("=" * 70)

print(f"Molecules: {len(df)}")


# ============================================================
# Tokenize all molecules
# ============================================================

token_counter = Counter()
token_lengths = []

example_tokens = []


for i, smiles in enumerate(
    df["canonical_smiles"]
):

    tokens = tokenize_smiles(smiles)
    token_counter.update(tokens)
    token_lengths.append(
        len(tokens)
    )

    if i < 10:
        example_tokens.append(
            {
                "smiles": smiles,
                "tokens": tokens,
            }
        )


# ============================================================
# Build vocabulary
#
# Special tokens come first so their IDs are fixed.
# Remaining tokens sorted alphabetically.
# ============================================================

chemical_tokens = sorted(
    token_counter.keys()
)


vocab = (
    SPECIAL_TOKENS
    + chemical_tokens
)


token_to_id = {
    token: idx
    for idx, token in enumerate(vocab)
}


id_to_token = {
    idx: token
    for token, idx in token_to_id.items()
}


# ============================================================
# IDs
# ============================================================

PAD_ID = token_to_id[PAD_TOKEN]
BOS_ID = token_to_id[BOS_TOKEN]
EOS_ID = token_to_id[EOS_TOKEN]
UNK_ID = token_to_id[UNK_TOKEN]


# ============================================================
# Sequence length
#
# +2 because later every sequence becomes:
#
# <BOS> tokens... <EOS>
# ============================================================

token_length_series = pd.Series(
    token_lengths
)

max_token_length = int(
    token_length_series.max()
)

max_sequence_length = (
    max_token_length + 2
)


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("VOCABULARY")
print("=" * 70)

print(
    f"Chemical tokens: {len(chemical_tokens)}"
)

print(
    f"Total vocabulary size: {len(vocab)}"
)

print()
print("Special token IDs:")

print(f"  PAD = {PAD_ID}")
print(f"  BOS = {BOS_ID}")
print(f"  EOS = {EOS_ID}")
print(f"  UNK = {UNK_ID}")


print()
print("Chemical vocabulary:")

for token in chemical_tokens:

    print(
        f"  {token!r:12s} "
        f"count={token_counter[token]}"
    )


print()
print("=" * 70)
print("TOKEN LENGTH")
print("=" * 70)

print(
    token_length_series.describe()
)

print()
print(
    f"Maximum token length:    "
    f"{max_token_length}"
)

print(
    f"Maximum sequence length "
    f"(BOS/EOS included): "
    f"{max_sequence_length}"
)


# ============================================================
# Example tokenizations
# ============================================================

print()
print("=" * 70)
print("TOKENIZATION EXAMPLES")
print("=" * 70)

for example in example_tokens:

    print()
    print(
        "SMILES:",
        example["smiles"],
    )

    print(
        "TOKENS:",
        example["tokens"],
    )


# ============================================================
# Save vocabulary
# ============================================================

vocab_data = {
    "token_to_id":
        token_to_id,

    "id_to_token":
        {
            str(k): v
            for k, v
            in id_to_token.items()
        },

    "special_tokens":
        {
            "PAD":
                PAD_TOKEN,

            "BOS":
                BOS_TOKEN,

            "EOS":
                EOS_TOKEN,

            "UNK":
                UNK_TOKEN,
        },

    "special_token_ids":
        {
            "PAD":
                PAD_ID,

            "BOS":
                BOS_ID,

            "EOS":
                EOS_ID,

            "UNK":
                UNK_ID,
        },

    "max_token_length":
        max_token_length,

    "max_sequence_length":
        max_sequence_length,
}


with open(
    OUTPUT_DIR / "smiles_vocab.json",
    "w",
) as f:

    json.dump(
        vocab_data,
        f,
        indent=4,
    )


# ============================================================
# Save token-frequency report
# ============================================================

token_report = pd.DataFrame(
    [
        {
            "token":
                token,

            "count":
                token_counter[token],

            "fraction":
                token_counter[token]
                / sum(token_counter.values()),
        }

        for token
        in chemical_tokens
    ]
)


token_report = (
    token_report
    .sort_values(
        "count",
        ascending=False,
    )
)


token_report.to_csv(
    REPORT_DIR
    / "smiles_token_frequency.csv",
    index=False,
)


# ============================================================
# Save tokenizer configuration
# ============================================================

summary = {
    "n_molecules":
        len(df),

    "vocab_size":
        len(vocab),

    "n_chemical_tokens":
        len(chemical_tokens),

    "max_token_length":
        max_token_length,

    "max_sequence_length":
        max_sequence_length,

    "mean_token_length":
        float(
            token_length_series.mean()
        ),
}


with open(
    REPORT_DIR
    / "tokenizer_summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


print()
print("=" * 70)
print("TOKENIZER COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(
    "  data/generation/processed/"
    "smiles_vocab.json"
)
print(
    "  reports/generation/"
    "smiles_token_frequency.csv"
)
print(
    "  reports/generation/"
    "tokenizer_summary.json"
)