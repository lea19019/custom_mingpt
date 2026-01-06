"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from mingpt.utils import CfgNode as CN

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # During warmup: linearly increase from 0 to base LR
            scale = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup: use base learning rate
            return self.base_lrs

class CosineAnnealingWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # During warmup: linearly increase from 0 to base LR
            scale = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup: cosine decay from base LR to min_lr
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            # Cosine decay formula: min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
            return [base_lr * scale for base_lr in self.base_lrs]

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.use_warmup = False
        C.warmup_steps = 1000
        C.use_cosine_scheduler = False
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            # Check if CUDA is available AND usable
            if torch.cuda.is_available():
                try:
                    # Test if CUDA is actually usable by allocating a small tensor
                    torch.zeros(1).cuda()
                    self.device = 'cuda'
                except (RuntimeError, torch.AcceleratorError):
                    # CUDA exists but is busy or unavailable
                    self.device = 'cpu'
                    print("Warning: CUDA devices are busy or unavailable, falling back to CPU")
            else:
                self.device = 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        if self.config.use_warmup and not self.config.use_cosine_scheduler:
            # print("Using Linear Warmup Scheduler")
            scheduler = LinearWarmupScheduler(
                self.optimizer, 
                warmup_steps=config.warmup_steps
            )
        if self.config.use_cosine_scheduler:
            # print("Using Cosine Annealing Warmup Scheduler")
            scheduler = CosineAnnealingWarmupScheduler(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                total_steps=config.max_iters
            )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=(self.device == 'cuda'),  # Only use pinned memory with CUDA
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            # print(batch)
            # print(batch[0])
            # print(batch[1])
            x, y = batch
            # print(x.size(), y.size())

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            if self.config.use_warmup or self.config.use_cosine_scheduler:
                # print("Stepping scheduler")
                scheduler.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
