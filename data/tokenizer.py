import torch 
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self,src_sentences,tgt_sentences,src_vocab,tgt_vocab,max_len=50):

        super().__init__()
        self.src_sentences=src_sentences
        self.tgt_sentences=tgt_sentences
        self.src_vocab=src_vocab
        self.tgt_vocab=tgt_vocab
        self.max_len=max_len
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, index):
        src=self.src_vocab.encode(self.src_sentences[index],self.max_len)
        tgt=self.tgt_vocab.encode(self.tgt_sentences[index],self.max_len)
        return torch.tensor(src),torch.tensor(tgt)
    