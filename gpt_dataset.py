from torch.utils.data import Dataset
import torch

"""
For organizing the data into chunks with input chunks and target chunks; input is what data is fed into GPT and target is what is the "correct" ans

Methods:
    :__init__: Takes in data and chunks them into input:answer chunks
        *params*: 
            txt - input training text
            tokenizer - which tokenizer to use
            max_length - size for context window for one dataset
            stride - jump size (How much to jump after dataset) #Avoids overfitting and faster
    :__len__: getter for length
    :__getitem__: getter for dictionary items
"""
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        # Sliding window to chunk the overlapping sequences by max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]