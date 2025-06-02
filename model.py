import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim) # shape -> (N, value_lengh, heads, head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) # shape -> (N, key_length, heads, head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim) # shape -> (N, query_length, heads, head_dim)

        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (0.5)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask) # in this case value, key, query are the same
        x = self.dropout(self.norm1(attention + query))  # Residual connection
        forward = self.ffn(x)
        out = self.dropout(self.norm2(forward + x))  # Residual connection
        return out


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        embed_size=256,
        num_layers=6,
        heads=8,
        dropout=0.1,
        forward_expansion=4,
        max_length=512,
    ):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.max_length = max_length

    def forward(self, x, mask=None):
        N, seq_length = x.shape # x is (batch_size, sequence_length)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        token_embeds = self.token_embedding(x) # (batch_size, sequence_length, embed_size)
        pos_embeds = self.position_embedding(positions) # (batch_size, sequence_length, embed_size)
        x = self.dropout(token_embeds + pos_embeds) # (batch_size, sequence_length, embed_size)

        for layer in self.layers:
            x = layer(x, x, x, mask)  # Self-attention: Q=K=V

        out = self.fc_out(x)
        """
        out shape is (batch_size, sequence_length, vocab_size)
        We return the output logits for each token in the sequence.
        The output can be used for next-token prediction or sequence generation by applying a softmax function to 
the logits. 
        """
        return out 

