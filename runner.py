import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, moves, tokenizer, chunk_size, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(moves)
        assert len(token_ids) > chunk_size, "Number of tokenized inputs must at least be equal to max_length+1"
        # Chunks the data into size max_length moving by stride spots every time
        for i in range(0, len(token_ids) - chunk_size, stride):
            input_chunk = token_ids[i:i + chunk_size]
            target_chunk = token_ids[i + 1: i + chunk_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, 
                         stride=4, shuffle=False, drop_last=True,
                         num_workers=0):

    # Create dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader