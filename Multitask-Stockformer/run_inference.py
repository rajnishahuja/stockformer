"""
Inference script to reproduce author's predictions using pre-trained model.
This script loads a saved model and generates predictions on the test set,
then compares with the author's saved predictions.
"""

import numpy as np
import torch
import argparse
import configparser
import math
import os
import pandas as pd
import re
from lib.Multitask_Stockformer_utils import log_string, metric, save_to_csv, StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help='configuration file')
parser.add_argument("--model_path", type=str, required=True, help='path to saved model')
parser.add_argument("--output_dir", type=str, required=True, help='directory to save predictions')
args_temp, unknown = parser.parse_known_args()

# Read configuration file
config = configparser.ConfigParser()
config.read(args_temp.config)

# Add configuration parameters
parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
parser.add_argument('--seed', type=int, default=config['train']['seed'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--Dataset', default=config['data']['dataset'])
parser.add_argument('--T1', type=int, default=config['data']['T1'])
parser.add_argument('--T2', type=int, default=config['data']['T2'])
parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])
parser.add_argument('--L', type=int, default=config['param']['layers'])
parser.add_argument('--h', type=int, default=config['param']['heads'])
parser.add_argument('--d', type=int, default=config['param']['dims'])
parser.add_argument('--j', type=int, default=config['param']['level'])
parser.add_argument('--s', type=float, default=config['param']['samples'])
parser.add_argument('--w', default=config['param']['wave'])
parser.add_argument('--traffic_file', default=config['file']['traffic'])
parser.add_argument('--indicator_file', default=config['file']['indicator'])
parser.add_argument('--adj_file', default=config['file']['adj'])
parser.add_argument('--adjgat_file', default=config['file']['adjgat'])

args = parser.parse_args()

# Setup device
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

def run_inference(model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat):
    """Run inference on test set"""
    model.eval()
    num_test = testXL.shape[0]
    num_batch = math.ceil(num_test / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    print(f"\nRunning inference on {num_test} samples ({num_batch} batches)...")
    
    with torch.no_grad():
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)

            xl = torch.from_numpy(testXL[start_idx:end_idx]).float().to(device)
            xh = torch.from_numpy(testXH[start_idx:end_idx]).float().to(device)
            xc = torch.from_numpy(testXC[start_idx:end_idx]).float().to(device)
            te = torch.from_numpy(testTE[start_idx:end_idx]).to(device)
            bonus = torch.from_numpy(bonus_testX[start_idx:end_idx]).float().to(device)
            y = testY[start_idx:end_idx]
            yc = testYC[start_idx:end_idx]

            hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

            pred_class.append(hat_y_class.cpu().numpy())
            pred_regress.append(hat_y_regress.cpu().numpy())
            label_class.append(yc)
            label_regress.append(y)
    
    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    print(f"Predictions generated:")
    print(f"  Classification shape: {pred_class.shape}")
    print(f"  Regression shape: {pred_regress.shape}")

    # Calculate metrics
    accs = []
    maes = []
    rmses = []
    mapes = []

    for i in range(pred_regress.shape[1]):
        acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :], 
                                     pred_class[:, i, :], label_class[:, i, :])
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    
    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    
    print(f"\nMetrics (averaged over all time steps):")
    print(f"  Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"  MAE:      {avg_mae:.6f}")
    print(f"  RMSE:     {avg_rmse:.6f}")
    print(f"  MAPE:     {avg_mape:.2f}%")

    # Save predictions
    os.makedirs(f"{args.output_dir}/classification", exist_ok=True)
    os.makedirs(f"{args.output_dir}/regression", exist_ok=True)
    
    save_to_csv(f'{args.output_dir}/classification/classification_pred_last_step.csv', pred_class[:, -1, :])
    save_to_csv(f'{args.output_dir}/classification/classification_label_last_step.csv', label_class[:, -1])
    save_to_csv(f'{args.output_dir}/regression/regression_pred_last_step.csv', pred_regress[:, -1, :])
    save_to_csv(f'{args.output_dir}/regression/regression_label_last_step.csv', label_regress[:, -1])
    
    print(f"\nPredictions saved to: {args.output_dir}")
    
    return pred_class, pred_regress, label_class, label_regress


def compare_predictions(new_output_dir, original_output_dir):
    """Compare newly generated predictions with author's saved predictions"""
    print(f"\n{'='*70}")
    print("COMPARING WITH AUTHOR'S SAVED PREDICTIONS")
    print(f"{'='*70}")
    
    # Compare regression predictions
    print("\n--- Regression Task ---")
    new_reg_pred = pd.read_csv(f'{new_output_dir}/regression/regression_pred_last_step.csv', header=None).values
    orig_reg_pred = pd.read_csv(f'{original_output_dir}/regression/regression_pred_last_step.csv', header=None).values
    
    diff = np.abs(new_reg_pred - orig_reg_pred)
    print(f"Shape - New: {new_reg_pred.shape}, Original: {orig_reg_pred.shape}")
    print(f"Max absolute difference: {diff.max():.10f}")
    print(f"Mean absolute difference: {diff.mean():.10f}")
    print(f"Predictions match: {np.allclose(new_reg_pred, orig_reg_pred, rtol=1e-5, atol=1e-8)}")
    
    # Compare classification predictions
    print("\n--- Classification Task ---")
    
    # Parse author's classification predictions
    pattern = r'\[\s*([-\d.]+)\s+([-\d.]+)\s*\]'
    orig_probs = []
    with open(f'{original_output_dir}/classification/classification_pred_last_step.csv', 'r') as f:
        for line in f:
            pairs = re.findall(pattern, line)
            row_probs = []
            for logit_down, logit_up in pairs:
                ld = float(logit_down)
                lu = float(logit_up)
                prob_up = np.exp(lu) / (np.exp(ld) + np.exp(lu))
                row_probs.append([ld, lu])
            orig_probs.append(row_probs)
    
    orig_probs = np.array(orig_probs)
    
    # Read new predictions (assuming same format)
    new_probs = []
    with open(f'{new_output_dir}/classification/classification_pred_last_step.csv', 'r') as f:
        for line in f:
            pairs = re.findall(pattern, line)
            row_probs = []
            for logit_down, logit_up in pairs:
                ld = float(logit_down)
                lu = float(logit_up)
                row_probs.append([ld, lu])
            new_probs.append(row_probs)
    
    new_probs = np.array(new_probs)
    
    print(f"Shape - New: {new_probs.shape}, Original: {orig_probs.shape}")
    
    if new_probs.shape == orig_probs.shape:
        diff_class = np.abs(new_probs - orig_probs)
        print(f"Max absolute difference: {diff_class.max():.10f}")
        print(f"Mean absolute difference: {diff_class.mean():.10f}")
        print(f"Predictions match: {np.allclose(new_probs, orig_probs, rtol=1e-5, atol=1e-8)}")
    else:
        print("WARNING: Shapes don't match, cannot compare directly")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    print("="*70)
    print("STOCKFORMER INFERENCE SCRIPT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.Dataset}")
    print(f"  Model path: {args_temp.model_path}")
    print(f"  Output directory: {args_temp.output_dir}")
    
    # Load test data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    test_dataset = StockDataset(args, mode='test')
    
    testXL = test_dataset.XL
    testXH = test_dataset.XH
    testXC = test_dataset.indicator_X
    testTE = test_dataset.TE
    testY = test_dataset.Y
    testYL = test_dataset.YL
    testYC = test_dataset.indicator_Y
    bonus_testX = test_dataset.bonus_X
    infeature = test_dataset.infea
    
    print(f"Test set loaded:")
    print(f"  Samples: {testXL.shape[0]}")
    print(f"  Time steps (T1): {testXL.shape[1]}")
    print(f"  Stocks: {testXL.shape[2]}")
    print(f"  Features: {infeature}")
    
    # Load graph
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)
    print(f"  Graph embeddings: {adjgat.shape}")
    
    # Construct model
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    outfea_class = 2
    outfea_regress = 1
    model = Stockformer(infeature, args.h*args.d, outfea_class, outfea_regress, 
                       args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    
    # Load pre-trained weights
    if os.path.exists(args_temp.model_path):
        model.load_state_dict(torch.load(args_temp.model_path, map_location=device))
        print(f"✓ Model loaded from: {args_temp.model_path}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    else:
        print(f"ERROR: Model file not found: {args_temp.model_path}")
        exit(1)
    
    # Run inference
    print(f"\n{'='*70}")
    print("RUNNING INFERENCE")
    print(f"{'='*70}")
    pred_class, pred_regress, label_class, label_regress = run_inference(
        model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat
    )
    
    # Compare with author's predictions if they exist
    original_output = args_temp.output_dir.replace('_reproduced', '')
    if os.path.exists(original_output):
        compare_predictions(args_temp.output_dir, original_output)
    else:
        print(f"\nNote: Original predictions not found at {original_output}")
        print("Skipping comparison step.")
    
    print("✓ Inference complete!")
