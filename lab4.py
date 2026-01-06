import torch
import os
from dataloader import JSONLDataset
from mingpt.model import GPT
from mingpt.trainer import Trainer

# FILE_PATH = '/nobackup/autodelete/usr/vacl2/pile_data_10_tiny_100.jsonl'
# FILE_PATH = '/nobackup/autodelete/usr/vacl2/pile_data_10_tiny.jsonl'
FILE_PATH = '/nobackup/autodelete/usr/vacl2/pile_data_10_first_50000.jsonl'
# FILE_PATH = '/nobackup/autodelete/usr/vacl2/pile_data_10.jsonl'

def create_batch_callback(loss_file, checkpoint_dir):
    """Factory function to create callback with specific loss file"""
    def batch_end_callback(trainer):
        if trainer.iter_num % 1000 == 0:
            loss_val = trainer.loss.item()
            print(f"iter {trainer.iter_num}: train loss {loss_val:.5f}")
            
            # Save loss to file
            with open(loss_file, 'a') as f:
                f.write(f"{trainer.iter_num},{loss_val}\n")
        
        # Save checkpoint every 25000 iterations
        if trainer.iter_num > 0 and trainer.iter_num % 25000 == 0:
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'iter_num': trainer.iter_num,
                'loss': trainer.loss.item()
            }
            filepath = os.path.join(checkpoint_dir, f'checkpoint_iter_{trainer.iter_num}.pt')
            torch.save(checkpoint, filepath)
            print(f"Saved checkpoint to {filepath}")
    
    return batch_end_callback

def main(config, experiment_name):
    # Create directories
    os.makedirs('./losses', exist_ok=True)
    checkpoint_dir = f'./checkpoints/{experiment_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    loss_file = f'./losses/{experiment_name}.csv'
    
    # Write header if file doesn't exist
    if not os.path.exists(loss_file):
        with open(loss_file, 'w') as f:
            f.write("iteration,loss\n")
    
    # Initialize dataset
    train_dataset = JSONLDataset(FILE_PATH, split='train', test_size=1000)
    test_dataset = JSONLDataset(FILE_PATH, split='test', test_size=1000)
    print(f'Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')
    
    print("Experiment:", experiment_name)
    print("Active:", [k for k, v in config.items() if v])
    
    # Model config
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model_config.merge_from_dict({
        'use_swiglu': config['use_swiglu'],
        'use_rope': config['use_rope'],
        'use_rmsnorm': config['use_rmsnorm']
    })
    model = GPT(model_config)

    # Training config
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-6
    train_config.max_iters = 175000
    train_config.batch_size = 16
    train_config.num_workers = 0
    train_config.merge_from_dict({
        'use_warmup': config['use_warmup'],
        'use_cosine_scheduler': config['use_cosine_scheduler'],
        'warmup_steps': 1000
    })
    
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', create_batch_callback(loss_file, checkpoint_dir))
    
    print(f"Starting training. Losses will be saved to {loss_file}")
    trainer.run()
    
    print(f"Training complete. Final checkpoint saved to {checkpoint_dir}")


# uv run python lab4.py -a
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')
    parser.add_argument('-s', '--use_swiglu', action='store_true')
    parser.add_argument('-r', '--use_rope', action='store_true')
    parser.add_argument('-n', '--use_rmsnorm', action='store_true')
    parser.add_argument('-w', '--use_warmup', action='store_true')
    parser.add_argument('-c', '--use_cosine_scheduler', action='store_true')
    
    args = parser.parse_args()
    
    # Determine experiment name
    if args.baseline:
        experiment_name = 'baseline'
        config = {k: False for k in ['use_swiglu', 'use_rope', 'use_rmsnorm', 'use_warmup', 'use_cosine_scheduler']}
    elif args.all:
        experiment_name = 'all_modifications'
        config = {k: True for k in ['use_swiglu', 'use_rope', 'use_rmsnorm', 'use_warmup', 'use_cosine_scheduler']}
    else:
        flags = []
        if args.use_swiglu: flags.append('swiglu')
        if args.use_rope: flags.append('rope')
        if args.use_rmsnorm: flags.append('rmsnorm')
        if args.use_warmup: flags.append('warmup')
        if args.use_cosine_scheduler: flags.append('cosine')
        
        experiment_name = '_'.join(flags) if flags else 'baseline'
        config = {
            'use_swiglu': args.use_swiglu,
            'use_rope': args.use_rope,
            'use_rmsnorm': args.use_rmsnorm,
            'use_warmup': args.use_warmup,
            'use_cosine_scheduler': args.use_cosine_scheduler,
        }
    
    main(config, experiment_name)