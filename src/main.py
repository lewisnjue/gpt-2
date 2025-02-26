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
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

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
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # Residual connection
        forward = self.ffn(x)
        out = self.dropout(self.norm2(forward + x))  # Residual connection
        return out


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
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
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        token_embeds = self.token_embedding(x)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)

        for layer in self.layers:
            x = layer(x, x, x, mask)  # Self-attention: Q=K=V

        out = self.fc_out(x)
        return out


""" traning setup """

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.seq_length]
        padding = [0] * (self.seq_length - len(tokens))
        tokens += padding
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])  # Shifted for next-token prediction

# Example tokenizer (use a real one like BPE in practice)
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        return [self.vocab.get(c, 0) for c in text.split()]  # Simple word-level tokenization

vocab = {"<pad>": 0, "I": 1, "love": 2, "machine": 3, "learning": 4, "Python": 5}
tokenizer = SimpleTokenizer(vocab)
texts = ["I love machine learning", "Python is great"]
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



""" traning loop """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(vocab_size=len(vocab), embed_size=128, num_layers=4, heads=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

train(model, dataloader, epochs=10)



""" text generation  """


def generate_text(model, tokenizer, start_text, max_length=20, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(tokens)
        logits = output[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == 0:  # Stop at padding
            break

    generated = tokens.squeeze().tolist()
    return " ".join([tokenizer.inverse_vocab.get(t, "") for t in generated])

print(generate_text(model, tokenizer, start_text="I love"))