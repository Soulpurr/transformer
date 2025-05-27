import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.transformer import Transformer
from utils.mask import create_masks


#mock data
def generate_dummy_data(batch_size,seq_len,vocab_size,pad_tokens=0):
    x=torch.randint(1,vocab_size,(batch_size,seq_len))
    y=x.clone()
    return x,y

#training loop

def train_model(model,optimizer,criterion,vocab_size,device,pad_token=0,epochs=10,batch_size=32,seq_len=20):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        x,y=generate_dummy_data(batch_size,seq_len,vocab_size)
        x,y=x.to(device),y.to(device)

        tgt_input=y[:,:-1]
        tgt_output=y[:,1:]
        #print(f"src shape: {x.shape}, tgt_input shape: {tgt_input.shape}")

        src_mask, tgt_mask, memory_mask = create_masks(x, tgt_input, pad_token)
        optimizer.zero_grad()
        output=model(x,tgt_input,src_mask,tgt_mask,memory_mask)

        loss=criterion(output.view(-1,vocab_size),tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
def main():
    vocab_size = 1000
    pad_token = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(vocab_size=vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    train_model(model, optimizer, criterion, vocab_size, device, pad_token)

if __name__ == "__main__":
    main()