"""
Analyze Stockformer model architecture and parameters
"""
import torch
import configparser
import argparse
from Stockformermodel.Multitask_Stockformer_models import Stockformer

# Parse config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/Multitask_Stock_Subset12.conf")
parser.add_argument("--model_path", type=str, default="cpt/STOCK/saved_model_Multitask_2020-12-02_2023-08-02")
args_temp = parser.parse_args()

config = configparser.ConfigParser()
config.read(args_temp.config)

# Model parameters
T1 = int(config['data']['T1'])
T2 = int(config['data']['T2'])
L = int(config['param']['layers'])
h = int(config['param']['heads'])
d = int(config['param']['dims'])
s = float(config['param']['samples'])

infeature = 363  # From test output
outfea_class = 2
outfea_regress = 1

print("="*80)
print("STOCKFORMER MODEL ARCHITECTURE ANALYSIS")
print("="*80)

# Create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Stockformer(infeature, h*d, outfea_class, outfea_regress, L, h, d, s, T1, T2, device).to(device)

# Load weights
model.load_state_dict(torch.load(args_temp.model_path, map_location=device))
model.eval()

print("\n" + "="*80)
print("MODEL CONFIGURATION")
print("="*80)
print(f"\nInput Parameters:")
print(f"  Input features (per stock):     {infeature}")
print(f"  Input time steps (T1):          {T1}")
print(f"  Output time steps (T2):         {T2}")
print(f"  Number of stocks:               255")

print(f"\nModel Hyperparameters:")
print(f"  Transformer layers (L):         {L}")
print(f"  Attention heads (h):            {h}")
print(f"  Embedding dimensions (d):       {d}")
print(f"  Hidden size (h*d):              {h*d}")
print(f"  Sparsity sampling ratio (s):    {s}")

print(f"\nOutput Parameters:")
print(f"  Classification output:          {outfea_class} classes (up/down)")
print(f"  Regression output:              {outfea_regress} value (return)")

# Analyze architecture
print("\n" + "="*80)
print("LAYER-BY-LAYER ARCHITECTURE")
print("="*80)

total_params = 0
trainable_params = 0

print(f"\n{'Layer Name':<50} {'Type':<25} {'Parameters':>15}")
print("-"*90)

for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Leaf modules only
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if num_params > 0:
            module_type = module.__class__.__name__
            print(f"{name:<50} {module_type:<25} {num_params:>15,}")
            total_params += num_params
            trainable_params += num_trainable

print("-"*90)
print(f"{'TOTAL':<50} {'':<25} {total_params:>15,}")
print(f"{'TRAINABLE':<50} {'':<25} {trainable_params:>15,}")

# Parameter breakdown by component
print("\n" + "="*80)
print("PARAMETER BREAKDOWN BY COMPONENT")
print("="*80)

component_params = {}
for name, param in model.named_parameters():
    # Extract component name (first part before .)
    component = name.split('.')[0] if '.' in name else name
    if component not in component_params:
        component_params[component] = 0
    component_params[component] += param.numel()

print(f"\n{'Component':<40} {'Parameters':>15} {'Percentage':>12}")
print("-"*70)
for component, num_params in sorted(component_params.items(), key=lambda x: x[1], reverse=True):
    percentage = (num_params / total_params) * 100
    print(f"{component:<40} {num_params:>15,} {percentage:>11.2f}%")

# Model size
print("\n" + "="*80)
print("MODEL SIZE")
print("="*80)
model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
print(f"\nTotal parameters:     {total_params:>15,}")
print(f"Model size (FP32):    {model_size_mb:>15,.2f} MB")
print(f"Model size on disk:   {model_size_mb:>15,.2f} MB (approx)")

# Input/Output shapes
print("\n" + "="*80)
print("INPUT/OUTPUT TENSOR SHAPES")
print("="*80)

print(f"\nInput tensors:")
print(f"  XL (low-freq):                  [batch, {T1}, stocks, {infeature}]")
print(f"  XH (high-freq):                 [batch, {T1}, stocks, {infeature}]")
print(f"  TE (time encoding):             [batch, {T1}, stocks, time_dim]")
print(f"  XC (trend indicator):           [batch, {T1}, stocks]")
print(f"  bonus_X (graph embeddings):     [batch, stocks, 128]")
print(f"  adjgat (graph structure):       [stocks, 128]")

print(f"\nOutput tensors:")
print(f"  Classification logits:          [batch, {T2}, stocks, {outfea_class}]")
print(f"  Classification last step:       [batch, stocks, {outfea_class}]")
print(f"  Regression predictions:         [batch, {T2}, stocks]")
print(f"  Regression last step:           [batch, stocks]")

# Architecture summary
print("\n" + "="*80)
print("ARCHITECTURE SUMMARY")
print("="*80)

print(f"""
The Stockformer model consists of:

1. WAVELET TRANSFORM MODULE
   - Decomposes input into low-frequency (trend) and high-frequency (noise)
   - Uses DWT1DForward with '{config['param']['wave']}' wavelet
   - Level: {config['param']['level']}

2. DUAL-FREQUENCY ENCODER
   - Separate pathways for low and high frequency components
   - Each pathway has {L} transformer layers
   - Multi-head self-attention with {h} head(s)
   - Dimension: {d} per head, total {h*d}

3. TEMPORAL & SPATIAL ATTENTION
   - Temporal attention: captures time dependencies
   - Sparse spatial attention: captures stock relationships
   - Sparsity ratio: {s} (keeps top {int(s*100)}% connections)

4. GRAPH-BASED FUSION
   - Uses graph embeddings (128-dim) from Struc2vec
   - Adaptive Graph Attention Network (GAT)
   - Fuses temporal and graph-based information

5. TCN (TEMPORAL CONVOLUTIONAL NETWORK)
   - Captures local temporal patterns
   - Dilated convolutions for larger receptive field

6. MULTI-TASK HEADS
   - Classification head: Binary trend prediction (up/down)
   - Regression head: Continuous return prediction
   - Shared representations, task-specific outputs

Total Parameters: {total_params:,}
Model Complexity: {'Low' if total_params < 500000 else 'Medium' if total_params < 2000000 else 'High'}
""")

print("="*80)
print("✓ Architecture analysis complete!")
print("="*80)
