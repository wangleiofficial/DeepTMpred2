"""
dataset processing
"""
from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import Dataset

class OPMDataset(Dataset):
    """OPM dataset
    """
    def __init__(self, seq_path):
        self.seq_record = SeqIO.index(seq_path, 'fasta')
        self.seq_keys = list(self.seq_record.keys())

    def __getitem__(self, index):
        _id = self.seq_keys[index]
        record = self.seq_record[_id]
        record_length = len(record.seq)
        seq_str = str(record.seq)
        return {"seq":(_id, seq_str), "seq_len":record_length}
            
    def __len__(self):
        return len(self.seq_keys)


def collate_fn(batch_record):
    data = [item['seq'] for item in batch_record]
    lengths = np.array([item['seq_len'] for item in batch_record])
    lengths = torch.from_numpy(lengths)
    return {"seq":data, "seq_len":lengths}

    



