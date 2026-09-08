#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SmilesVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)

        # Encoder
        self.encoder_gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        # Encoder hidden state -> latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ====================================================
        # Decoder
        # Each decoder step receives: token embedding + latent vector z
        # ====================================================
        self.decoder_gru = nn.GRU(
            input_size=embedding_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0))

        # z initializes decoder hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        # Decoder hidden -> vocabulary logits
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    # Encoder
    def encode(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        # pack_padded_sequence prevents the GRU from
        # processing PAD tokens as real sequence content.

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder_gru(packed)

        # Last GRU layer
        final_hidden = hidden[-1]
        mu = self.fc_mu(final_hidden)
        logvar = self.fc_logvar(final_hidden)
        return mu, logvar

    # Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = (mu + epsilon * std)
        return z

    def decode(self, decoder_input_ids,z):
        embedded = self.embedding(decoder_input_ids)
        # Repeat latent vector at every timestep
        z_expanded = (z.unsqueeze(1).expand(-1, embedded.size(1), -1,))
        decoder_input = torch.cat([embedded, z_expanded], dim=-1)

        # Initialize GRU hidden state using z
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(z.size(0), self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()

        decoder_output, _ = (self.decoder_gru(decoder_input, hidden))
        logits = self.output_layer(decoder_output)
        return logits

    # Full forward pass
    def forward(self, input_ids, lengths, decoder_input_ids):
        mu, logvar = self.encode(input_ids, lengths)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(decoder_input_ids, z)
        
        return (logits, mu, logvar, z,)