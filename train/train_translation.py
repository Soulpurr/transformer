import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import Transformer
from utils.mask import create_masks
from data.translation_dataset import TranslationDataset
from data.vocab import Vocab
import pandas as pd
import numpy as np

# --- Accuracy computation ---
def compute_accuracy(preds, targets, pad_token):
    preds = preds.argmax(dim=-1)
    mask = targets != pad_token
    correct = (preds == targets) & mask
    return correct.sum().item(), mask.sum().item()

# --- Training loop ---
def train_model(model, loader, optimizer, criterion, device, pad_token, tgt_vocab_size, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input, pad_token)

            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
            loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            correct, tokens = compute_accuracy(output, tgt_output, pad_token)
            total_correct += correct
            total_tokens += tokens

        avg_loss = total_loss / len(loader)
        avg_acc = total_correct / total_tokens
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Accuracy = {avg_acc * 100:.2f}%")

# --- Evaluation loop ---
@torch.no_grad()
def evaluate_model(model, loader, criterion, device, pad_token, tgt_vocab_size):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input, pad_token)

        output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
        loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))
        total_loss += loss.item()

        correct, tokens = compute_accuracy(output, tgt_output, pad_token)
        total_correct += correct
        total_tokens += tokens

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_tokens
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc * 100:.2f}%")


# --- Main driver ---
def main():
    # Load dataset
    df = pd.read_csv("./data/eng_-french.csv").dropna()
    df = df.sample(5000)
    src_sentences = np.array(df["English words/sentences"])
    tgt_sentences = np.array(df["French words/sentences"])

    # Build vocabulary
    src_vocab = Vocab(src_sentences)
    tgt_vocab = Vocab(tgt_sentences)

    # Split for train/val
    split_idx = 200
    train_dataset = TranslationDataset(src_sentences[:split_idx], tgt_sentences[:split_idx], src_vocab, tgt_vocab, max_len=20)
    val_dataset   = TranslationDataset(src_sentences[split_idx:], tgt_sentences[split_idx:], src_vocab, tgt_vocab, max_len=20)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(vocab_size=len(tgt_vocab))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi["<pad>"])

    # Train
    train_model(model, train_loader, optimizer, criterion, device, src_vocab.stoi["<pad>"], len(tgt_vocab), epochs=10)

    # Evaluate
    evaluate_model(model, val_loader, criterion, device, src_vocab.stoi["<pad>"], len(tgt_vocab))


if __name__ == "__main__":
    main()
