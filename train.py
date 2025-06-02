from model import GPT2 
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
import pandas as pd
from pathlib import Path


data_path = Path("data.txt")

with open(data_path, "r") as f:
    data = f.read()


class TextDataset(Dataset):
    def __init__(self, text, seq_length=64, pad_token='[PAD]'):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text) | set([pad_token]))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.pad_token = pad_token
        self.pad_idx = self.char_to_idx[pad_token]
        # Only use full-length sequences
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    import os
    import tqdm
    import torch
    from torch.utils.data import DataLoader

    # Hyperparameters
    seq_length = 64  # Reduce sequence length
    batch_size = 32  # Reduce batch size
    num_epochs = 20
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "gpt2_char_model.pt"

    # Prepare dataset and dataloader
    dataset = TextDataset(data, seq_length=seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model
    model = GPT2(vocab_size=dataset.vocab_size, max_length=seq_length, embed_size=128, num_layers=2, heads=2, dropout=0.1, forward_expansion=2)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # (batch, seq_length, vocab_size)
            # We want to predict the next character, so use only the last position
            logits = logits[:, -1, :]  # (batch, vocab_size)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'vocab': dataset.vocab,
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char
        }, model_save_path)

        # Evaluation after 10 epochs
        if epoch % 10 == 0:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for x, y in tqdm.tqdm(train_loader, desc="Evaluating"):
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    logits = logits[:, -1, :]
                    loss = criterion(logits, y)
                    eval_loss += loss.item() * x.size(0)
            eval_loss /= len(dataset)
            print(f"Evaluation Loss after {epoch} epochs: {eval_loss:.4f}")