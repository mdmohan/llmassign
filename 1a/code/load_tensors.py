import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MemmapDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int = 1024):
        self.seq_len = seq_len
        # Memory-map binary file (read-only)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_samples = len(self.data) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # Pull slice from disk and convert directly to PyTorch LongTensor
        # Fast zero-copy tensor creation from numpy buffer + cast to long
        chunk = torch.from_numpy(self.data[start:end]).long()
        
        # Return input_ids and labels without unnecessary cloned allocations
        return {"input_ids": chunk, "labels": chunk}


def load_tokens_from_bin(
    filename: str, 
    context_length: int = 1024, 
    batch_size: int = 8, 
    num_workers: int = 4
) -> DataLoader:
    dataset = MemmapDataset(filename, seq_len=context_length)
    print(f"Total dataset samples: {len(dataset)}")
    print(f"Sample tensor shape:   {dataset[0]['input_ids'].shape}")  # torch.Size([1024])
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,        # Set to False if dataset is huge and disk I/O becomes bottleneck
        num_workers=num_workers, # Multi-process loading to eliminate GPU starvation
        pin_memory=True,     # Speeds up CPU -> GPU transfers
        # Every packed sequence is valid. Keep the final smaller batch rather
        # than silently discarding up to batch_size - 1 sequences each epoch.
        drop_last=False,
    )
    
    return train_dataloader


if __name__ == "__main__":
    # Example test run
    bin_path = "data/tokens.bin"
    context_length = 1024
    batch_size = 8
    
    dataloader = load_tokens_from_bin(
        filename=bin_path, 
        context_length=context_length, 
        batch_size=batch_size,
        num_workers=4
    )
    
    # Inspection loop
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"] # [8, 1024]
        labels = batch["labels"]       # [8, 1024]
        
        print(f"Batch {step + 1} Shapes:")
        print(f"  input_ids: {input_ids.shape} | dtype: {input_ids.dtype}")
        print(f"  labels:    {labels.shape} | dtype: {labels.dtype}")
        
        break
