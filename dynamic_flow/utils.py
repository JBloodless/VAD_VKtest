import math
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, dropout):
        super(TransformerEncoder, self).__init__()

        self.layers: List = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x, sources_mask=None, sources_key_padding_mask=None):
        # x: (batch_size, source_length, d_model)
        print(x.shape, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        for layer_index, layer in enumerate(self.layers):
            x = layer(
                x,
                sources_mask=sources_mask,
                sources_key_padding_mask=sources_key_padding_mask,
            )
        x = self.layer_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.self_attention_sublayer = Sublayer(d_model=d_model, dropout=dropout)

        self.feed_forward = PositionwiseFeedForwardNetwork(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        self.feed_forward_sublayer = Sublayer(d_model=d_model, dropout=dropout)

    def forward(self, x, sources_mask=None, sources_key_padding_mask=None):
        print(x.shape, '!!!!!!!!!!')
        x, _ = self.self_attention_sublayer(
            x,
            lambda x: self.self_attention(
                query=x,
                key=x,
                value=x,
                attention_mask=sources_mask,
                key_padding_mask=sources_key_padding_mask,
            ),
        )
        x, _ = self.feed_forward_sublayer(x, self.feed_forward)
        return


class Sublayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(Sublayer, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        print(x.shape)
        # Residual connection to unnormalized value. Different from paper.
        sublayer_x, state = sublayer(self.layer_norm(x))
        x = self.dropout(sublayer_x) + x
        return x, state


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, cache_mode=None):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, n_heads * self.d_head)
        self.key_projection = nn.Linear(d_model, n_heads * self.d_head)
        self.value_projection = nn.Linear(d_model, n_heads * self.d_head)
        self.final_projection = nn.Linear(d_model, n_heads * self.d_head)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=3)

        self.cache_mode = cache_mode

    def forward(
            self,
            query,
            key,
            value,
            attention_mask=None,
            key_padding_mask=None,
            cache=None,
    ):
        """
        Args:
            query: (batch_size, query_length, d_model)
            key: (batch_size, key_length, d_model)
            value: (batch_size, value_length, d_model)
            attention_mask: (query_length, key_length)
            key_padding_mask: (batch_size, key_length)
            cache
        """
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.n_heads

        query_projected = self.query_projection(query)
        if not cache or cache["key_projected"] is None or cache["value_projected"] is None:
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:
            if self.cache_mode == "self-attention":
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)
                key_projected = torch.cat([cache["key_projected"], key_projected], dim=1)
                value_projected = torch.cat([cache["value_projected"], value_projected], dim=1)
            elif self.cache_mode == "memory-attention":
                key_projected = cache["key_projected"]
                value_projected = cache["value_projected"]
            else:
                raise NotImplementedError

        if self.cache_mode is not None:
            cache = {"key_projected": key_projected, "value_projected": value_projected}
        else:
            cache = None

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.n_heads, d_head).transpose(
            1, 2
        )
        # query_heads: (batch_size, n_heads, query_len, d_head)
        key_heads = key_projected.view(batch_size, key_len, self.n_heads, d_head).transpose(1, 2)
        # key_heads: (batch_size, n_heads, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.n_heads, d_head).transpose(
            1, 2
        )
        # value_heads: (batch_size, n_heads, value_len, d_head)

        attention_scores = self.scaled_dot_product(query_heads, key_heads)
        # attention_scores: (batch_size, n_heads, query_len, key_len)

        if key_padding_mask is not None:
            key_padding_mask_expanded = (
                key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=1).expand_as(attention_scores)
            )
            attention_scores = attention_scores.masked_fill(
                key_padding_mask_expanded, float("-inf")
            )
        if attention_mask is not None:
            attention_mask_expanded = (
                attention_mask.unsqueeze(dim=0).unsqueeze(dim=0).expand_as(attention_scores)
            )
            attention_mask_expanded = attention_mask_expanded.to(dtype=attention_scores.dtype)
            attention_scores = attention_scores + attention_mask_expanded

        attention = self.softmax(attention_scores)
        # attention: (batch_size, n_heads, query_len, key_len)

        attention_dropped = self.dropout(attention)

        context_heads = torch.matmul(
            attention_dropped, value_heads
        )  # (batch_size, n_heads, query_len, d_head)
        context_sequence = context_heads.transpose(
            1, 2
        ).contiguous()  # (batch_size, query_len, n_heads, d_head)
        context = context_sequence.view(
            batch_size, query_len, d_model
        )  # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)

        return final_output, (cache, attention)

    def scaled_dot_product(self, query_heads, key_heads):
        """
        Args:
             query_heads: (batch_size, n_heads, query_len, d_head)
             key_heads: (batch_size, n_heads, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(
            query_heads, key_heads_transposed
        )  # (batch_size, n_heads, query_len, key_len)
        attention_scores = dot_product / np.sqrt(self.d_head)
        return attention_scores


class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x), None


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, encoding_size, initial_length=100):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.encoding_size = encoding_size
        self.scale = math.sqrt(encoding_size)  # To make embeddings relatively larger
        self.pe = self.build_positional_encoding(length=initial_length)

    def forward(self, x):
        # x: (batch_size, sequenceh_length, dimension=encoding_size)
        if x.size(1) > self.pe.size(1):
            del self.pe
            torch.cuda.empty_cache()
            self.pe = self.build_positional_encoding(length=x.size(1))
        if self.pe.dtype != x.dtype or self.pe.device != x.device:
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)

        return x + self.pe[:, : x.size(1)] / self.scale

    def build_positional_encoding(self, length):
        # Compute the positional encodings once in log space.
        pe = torch.zeros(length, self.encoding_size)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.encoding_size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.encoding_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
