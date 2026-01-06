import json
import torch
from torch.utils.data import Dataset
from mingpt.bpe import BPETokenizer

PAD_TOKEN = 50256

class JSONLDataset(Dataset):
    def __init__(self, file_path, split='train', test_size=1000, block_size=512):
        self.block_size = block_size
        # Initialize BPE tokenizer (downloads to ~/.cache/mingpt/ on first run)
        self.tokenizer = BPETokenizer()
        self.file_path = file_path
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # print(f'Length of lines: {len(lines)}')
        # prelength = len(lines)
        data = []
        for i, line in enumerate(lines):
            try:
                # Processes the line and extracts the 'text' field
                d = json.loads(line)['text']
                # Skips short text entries
                if len(d) < 10:
                    continue
                data.append(d)
            except json.JSONDecodeError:
                # Silently skip lines that fail to parse
                pass 

        # print(f'Length of data: {len(data) / prelength * 100:.2f}%')
        
        # Implements the train/test split
        if split == 'train':
            self.data = data[:-test_size]
        else:
            self.data = data[-test_size:]

    def __len__(self):
        # Returns the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # print(text)
        # Tokenize on-demand
        tokens = self.tokenizer(text)  # Returns tensor with shape [1, seq_len]
        tokens = tokens.squeeze(0)      # Remove batch dimension -> [seq_len]

        # We need block_size + 1 tokens to create input and target pairs
        if len(tokens) > self.block_size + 1:
            tokens = tokens[:self.block_size + 1]

        # Pad if too short
        if len(tokens) < self.block_size + 1:
            padding = torch.full((self.block_size + 1 - len(tokens),), PAD_TOKEN, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])

        # x = input tokens, y = target tokens (shifted by 1 for next-token prediction)
        x = tokens[:-1]  # All tokens except the last
        y = tokens[1:]   # All tokens except the first
        # print(tokens.size(), x.size(), y.size())
        # print(tokens)
        return x, y
    
    def get_vocab_size(self):
        return 50257
    
    def get_block_size(self):
        return self.block_size
