import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create visualizations folder if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Read all loss files
experiments = ['baseline', 'swiglu', 'rope', 'rmsnorm', 'warmup', 'cosine', 'all_modifications']
colors = ['black', 'blue', 'green', 'red', 'orange', 'purple', 'cyan']

# Load data for all experiments (None if missing)
data = {}
for exp in experiments:
    file_path = f'losses/{exp}.csv'
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            data[exp] = df
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            data[exp] = None
    else:
        data[exp] = None

# Create individual plots for each experiment (one PNG per experiment)
for exp, color in zip(experiments, colors):
    df = data.get(exp)
    save_path = f'visualizations/{exp}.png'
    plt.figure(figsize=(10, 6))
    if df is not None and 'iteration' in df.columns and 'loss' in df.columns:
        plt.plot(df['iteration'], df['loss'], color=color, linewidth=2, alpha=0.8)
        plt.xlabel('Training Steps', fontsize=14)
        plt.ylabel('NLL Loss', fontsize=14)
        plt.title(f'Training Loss - {exp}', fontsize=16)
        plt.grid(True, alpha=0.3)
    else:
        # Show an informative placeholder if the data is missing
        plt.text(0.5, 0.5, 'Missing or invalid data', ha='center', va='center', fontsize=14)
        plt.title(f'Training Loss - {exp} (no data)', fontsize=16)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved individual plot to {save_path}")

# Create combined comparison plot (all experiments overlaid)
plt.figure(figsize=(12, 8))
for exp, color in zip(experiments, colors):
    df = data.get(exp)
    if df is not None and 'iteration' in df.columns and 'loss' in df.columns:
        plt.plot(df['iteration'], df['loss'], label=exp, color=color, linewidth=2, alpha=0.8)

plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('NLL Loss', fontsize=14)
plt.title('Training Loss Comparison Across Modifications', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/training_comparison.png', dpi=300)
plt.close()
print("Saved combined plot to visualizations/training_comparison.png")

# Create a single large PNG with 7 separate subplots (3x3 grid, 2 empty slots turned off)
n = len(experiments)
rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(18, 14))
axes_flat = axes.flatten()

for i, (exp, color) in enumerate(zip(experiments, colors)):
    ax = axes_flat[i]
    df = data.get(exp)
    if df is not None and 'iteration' in df.columns and 'loss' in df.columns:
        ax.plot(df['iteration'], df['loss'], color=color, linewidth=2, alpha=0.9)
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('NLL Loss', fontsize=11)
        ax.set_title(exp, fontsize=12)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, 'Missing or invalid data', ha='center', va='center', fontsize=12)
        ax.set_title(f'{exp} (no data)', fontsize=12)
        ax.axis('off')

# Turn off any remaining unused axes
for j in range(n, rows * cols):
    axes_flat[j].axis('off')

plt.tight_layout()
grid_save_path = 'visualizations/individual_grid.png'
fig.savefig(grid_save_path, dpi=300)
plt.close(fig)
print(f"Saved grid plot to {grid_save_path}")