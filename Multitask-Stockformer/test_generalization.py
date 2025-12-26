"""
Test model generalization across different time periods
"""
import subprocess
import os
import numpy as np
import pandas as pd
import re

# Test subsets with their names
test_subsets = [
    ("Subset 1", "2018-03-01_2020-10-29", "Much earlier period"),
    ("Subset 4", "2018-11-28_2021-07-28", "Earlier period"),
    ("Subset 9", "2020-10-29_2021-07-28", "Earlier, Overlaps"),
    ("Subset 10", "2021-07-28_2022-08-02", "Similar Timeframe"),
    ("Subset 13", "2023-08-02_2023-08-02", "Later, Extends"),
]

# Model trained on Subset 12: 2020-12-02 to 2023-08-02
model_path = "cpt/STOCK/saved_model_Multitask_2020-12-02_2023-08-02"
python_exe = "/home/ubuntu/miniconda3/envs/phi3-k8s-env/bin/python"

print("="*80)
print("MODEL GENERALIZATION TEST")
print("="*80)
print(f"\nModel: Trained on Subset 12 (2020-12-02 to 2023-08-02)")
print(f"Testing on {len(test_subsets)} different time periods...\n")

results = []

for subset_name, date_range, description in test_subsets:
    print(f"\n{'='*80}")
    print(f"Testing on {subset_name}: {date_range}")
    print(f"Description: {description}")
    print(f"{'='*80}")
    
    # Create config file for this subset
    config_file = f"config/Multitask_Stock_{date_range}.conf"
    output_dir = f"output/Generalization_test_{date_range}"
    
    config_content = f"""[file]
traffic = ./data/gdrive_folder/Stock_CN_{date_range}/flow.npz
indicator = ./data/gdrive_folder/Stock_CN_{date_range}/trend_indicator.npz
adj = ./data/gdrive_folder/Stock_CN_{date_range}/corr_adj.npy
adjgat = ./data/gdrive_folder/Stock_CN_{date_range}/128_corr_struc2vec_adjgat.npy
model = {model_path}
log = ./log/STOCK/log_generalization_{date_range}

[data]
dataset = STOCK
T1 = 20
T2 = 2
train_ratio = 0.75
val_ratio = 0.125
test_ratio = 0.125

[train]
cuda = 0
max_epoch = 100
batch_size = 12
learning_rate = 0.001
seed = 1

[param]
layers = 2
heads = 1
dims = 128
samples = 1
wave = sym2
level = 1
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Run inference
    cmd = [
        python_exe,
        "run_inference.py",
        "--config", config_file,
        "--model_path", model_path,
        "--output_dir", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse metrics from output
        output = result.stdout
        
        # Extract metrics
        acc_match = re.search(r'Accuracy: ([\d.]+)', output)
        mae_match = re.search(r'MAE:\s+([\d.]+)', output)
        rmse_match = re.search(r'RMSE:\s+([\d.]+)', output)
        
        if acc_match and mae_match and rmse_match:
            acc = float(acc_match.group(1))
            mae = float(mae_match.group(1))
            rmse = float(rmse_match.group(1))
            
            results.append({
                'subset': subset_name,
                'date_range': date_range,
                'description': description,
                'accuracy': acc,
                'mae': mae,
                'rmse': rmse,
                'status': 'Success'
            })
            
            print(f"\n✓ Results:")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  MAE:      {mae:.6f}")
            print(f"  RMSE:     {rmse:.6f}")
        else:
            results.append({
                'subset': subset_name,
                'date_range': date_range,
                'description': description,
                'accuracy': None,
                'mae': None,
                'rmse': None,
                'status': 'Failed to parse'
            })
            print(f"\n✗ Failed to parse metrics")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
    
    except subprocess.TimeoutExpired:
        results.append({
            'subset': subset_name,
            'date_range': date_range,
            'description': description,
            'accuracy': None,
            'mae': None,
            'rmse': None,
            'status': 'Timeout'
        })
        print(f"\n✗ Timeout")
    except Exception as e:
        results.append({
            'subset': subset_name,
            'date_range': date_range,
            'description': description,
            'accuracy': None,
            'mae': None,
            'rmse': None,
            'status': f'Error: {str(e)}'
        })
        print(f"\n✗ Error: {e}")

# Print summary
print(f"\n\n{'='*80}")
print("GENERALIZATION TEST SUMMARY")
print(f"{'='*80}\n")

# Baseline (Subset 12 - training data)
print(f"{'Subset':<12} {'Date Range':<25} {'Accuracy':<12} {'MAE':<12} {'RMSE':<12}")
print(f"{'-'*80}")
print(f"{'Subset 12':<12} {'2020-12-02_2023-08-02':<25} {'53.79%':<12} {'0.014468':<12} {'0.020668':<12} (Baseline)")

for r in results:
    acc_str = f"{r['accuracy']*100:.2f}%" if r['accuracy'] else "N/A"
    mae_str = f"{r['mae']:.6f}" if r['mae'] else "N/A"
    rmse_str = f"{r['rmse']:.6f}" if r['rmse'] else "N/A"
    print(f"{r['subset']:<12} {r['date_range']:<25} {acc_str:<12} {mae_str:<12} {rmse_str:<12}")

print(f"\n{'='*80}")
print("CONCLUSIONS:")
successful = [r for r in results if r['status'] == 'Success']
if successful:
    accs = [r['accuracy'] for r in successful]
    maes = [r['mae'] for r in successful]
    
    print(f"  - Accuracy range: {min(accs)*100:.2f}% to {max(accs)*100:.2f}%")
    print(f"  - MAE range: {min(maes):.6f} to {max(maes):.6f}")
    
    # Compare to baseline
    baseline_acc = 0.5379
    baseline_mae = 0.014468
    
    acc_diffs = [(abs(a - baseline_acc)/baseline_acc)*100 for a in accs]
    mae_diffs = [(abs(m - baseline_mae)/baseline_mae)*100 for m in maes]
    
    print(f"  - Avg accuracy deviation from baseline: {np.mean(acc_diffs):.2f}%")
    print(f"  - Avg MAE deviation from baseline: {np.mean(mae_diffs):.2f}%")
    
    if np.mean(acc_diffs) < 10 and np.mean(mae_diffs) < 20:
        print(f"\n  ✓ Model shows GOOD generalization across time periods!")
    elif np.mean(acc_diffs) < 20 and np.mean(mae_diffs) < 40:
        print(f"\n  ~ Model shows MODERATE generalization")
    else:
        print(f"\n  ✗ Model shows POOR generalization")
else:
    print(f"  ✗ No successful tests completed")

print(f"{'='*80}\n")
