from collections import Counter

class Vocab:
    def __init__(self,sentences,min_freq=2,max_size=10000):
        self.pad_token="<pad>"
        self.sos_token="<sos>"
        self.eos_token="<eos>"
        self.unk_token="<unk>"

        tokens=[token for sent in sentences for token in sent.lower().split()]
        freq=Counter(tokens)
        vocab=[self.pad_token,self.sos_token,self.eos_token,self.unk_token] + [word for word,count in freq.items() if count>=min_freq][:max_size]
        self.stoi={word:idx for idx,word in enumerate(vocab)}
        self.itos={idx:word for word,idx in self.stoi.items()}
    def encode(self,sentence,max_len):
        tokens=sentence.lower().split()
        tokens=[self.sos_token]+tokens[:max_len-2]+[self.eos_token]
        ids=[self.stoi.get(token,self.stoi[self.unk_token]) for token in tokens]
        padding=[self.stoi[self.pad_token]]*(max_len-len(ids))
        return ids+padding
    def __len__(self):
        return len(self.stoi)