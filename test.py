import torch
from mingpt.model import GPT
from dataloader import JSONLDataset

FILE_PATH = '/nobackup/autodelete/usr/vacl2/pile_data_10_first_50000.jsonl'

# Test with smallest config
config = GPT.get_default_config()
config.model_type = 'gpt2'
config.vocab_size = 50257
config.block_size = 512
config.use_swiglu = True
config.use_rope = True
config.use_rmsnorm = True

model = GPT(config).cuda()
print(f"Model created. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Create one batch
dataset = JSONLDataset(FILE_PATH, split='train', test_size=10)
x, y = dataset[0]
x = x.unsqueeze(0).repeat(8, 1).cuda()  # batch_size=8
y = y.unsqueeze(0).repeat(8, 1).cuda()

print(f"Data loaded. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Forward pass
logits, loss = model(x, y)
print(f"After forward. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Backward pass
loss.backward()
print(f"After backward. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")