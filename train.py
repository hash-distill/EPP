import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import os
import csv
import math
from train_model import BiLSTM
from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
except ModuleNotFoundError:
    BinaryAUROC = None
    BinaryAveragePrecision = None


def accuracy(y_true, y_prob, device, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    correct = (y_pred == y_true).float().sum()
    return (correct / y_true.numel()).item()

def calculate_axis_accuracy(y_true, y_prob, device, threshold=0.5):
    # Stub metric returning 0 for simplicity if exact axis-wise accuracy isn't defined
    return 0.0, 0.0

def _safe_div(num, denom):
    return num / denom if denom != 0 else 0.0

def _compute_confusion(y_true, y_prob, threshold=0.5):
    y_true = y_true.detach().view(-1)
    y_prob = y_prob.detach().view(-1)
    y_pred = y_prob >= threshold
    y_true_pos = y_true >= 0.5

    tp = (y_pred & y_true_pos).sum().item()
    fp = (y_pred & ~y_true_pos).sum().item()
    tn = (~y_pred & ~y_true_pos).sum().item()
    fn = (~y_pred & y_true_pos).sum().item()
    return tp, fp, tn, fn

def _metrics_from_confusion(tp, fp, tn, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div((tp * tn - fp * fn), denom) if denom != 0 else 0.0
    return precision, recall, f1, mcc

def _safe_roc_auc(y_true_flat, y_prob_flat):
    # y_true_flat: {0,1}, y_prob_flat: [0,1]
    # roc_auc_score requires both classes exist; otherwise return nan.
    y_true_flat = np.asarray(y_true_flat).astype(int)
    y_prob_flat = np.asarray(y_prob_flat).astype(float)
    if len(np.unique(y_true_flat)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true_flat, y_prob_flat))

def _safe_auprc(y_true_flat, y_prob_flat):
    y_true_flat = np.asarray(y_true_flat).astype(int)
    y_prob_flat = np.asarray(y_prob_flat).astype(float)
    if len(np.unique(y_true_flat)) < 2:
        return float('nan')
    return float(average_precision_score(y_true_flat, y_prob_flat))

class Config:
    hidden_size = 256
    dropout = 0.2
    layer = 1
    learning_rate = 1e-3
    epoch = 100
    batch_size = 32
    pos_weight = 2
    threshold = 0.5
    device = "cuda:0"

def _resolve_data_path(path_value, default_value, repo_root):
    path = Path(path_value) if path_value else Path(default_value)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def main(device_override=None, ag_path=None, ab_path=None, label_path=None):
    '''Training the final EPP model'''
    ### 0. Check device (force GPU)
    device_setting = device_override if device_override is not None else Config.device
    if device_setting is None or device_setting == "auto":
        device_setting = "cuda:0"

    device = torch.device(device_setting)
    if device.type != "cuda":
        raise ValueError(f"CPU is forbidden. Please use a CUDA device, got: {device_setting}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU execution.")

    ### 1. Load Input Data and Label Y
    repo_root = Path(__file__).resolve().parent
    ag_path = _resolve_data_path(ag_path, "data/traindata_esm_ag.pt", repo_root)
    ab_path = _resolve_data_path(ab_path, "data/traindata_esm_ab.pt", repo_root)
    label_path = _resolve_data_path(label_path, "data/label_y.pt", repo_root)

    train_data_ag = torch.load(str(ag_path))
    train_data_ab = torch.load(str(ab_path))
    train_data_y = torch.load(str(label_path))

    # Partition data set
    test_size = 0.2
    X_train_ag, X_val_ag, X_train_ab, X_val_ab, y_train, y_val = train_test_split(train_data_ag, train_data_ab, train_data_y, test_size=test_size, random_state=42)
    print("Data read successfully.")

    ### 2. Define Model
    input_size = 1280  # Input feature dimension
    hidden_size = Config.hidden_size 
    dropout_rate = Config.dropout
    num_layers= Config.layer

    # model = BiLSTMMerge(input_size, hidden_size, num_layers, dropout_rate).to(device)
    model_ag = BiLSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)
    model_ab = BiLSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)
    
    # Define loss functions and optimizers
    pos_weight = torch.tensor([Config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(list(model_ag.parameters()) + list(model_ab.parameters()), lr=Config.learning_rate)
    print("Model construction Successful.")


    ### 3. Train Model
    num_epochs = Config.epoch
    batch_size = Config.batch_size
    threshold = Config.threshold
    print("Start model training")
    
    
    # Convert the Dataset into a PyTorch Dataset object
    train_dataset = TensorDataset(X_train_ag, X_train_ab, y_train)
    val_dataset = TensorDataset(X_val_ag, X_val_ab, y_val)
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, 'train_metrics.csv')

    model_ag, model_ab = train_model(
        train_loader, val_loader, model_ag, model_ab, optimizer, criterion, num_epochs, batch_size, threshold, device,
        pin_memory=pin_memory,
        metrics_path=metrics_path)

    # After training: plot curves from metrics CSV
    try:
        import pandas as _pd
        import matplotlib.pyplot as _plt

        df = _pd.read_csv(metrics_path)
        # last row for each split is the summary
        train_last = df[df['split'] == 'train'].iloc[-1]
        val_last = df[df['split'] == 'val'].iloc[-1]

        summary_path = os.path.join(run_dir, 'summary_metrics.csv')
        summary_df = _pd.DataFrame([
            {'split': 'train', 'loss': train_last['loss'], 'accuracy': train_last['accuracy'],
             'precision': train_last['precision'], 'recall': train_last['recall'], 'f1': train_last['f1'], 'mcc': train_last['mcc'],
             'auroc': train_last.get('auroc', float('nan')), 'auprc': train_last.get('auprc', float('nan')),
             'tp': train_last['tp'], 'fp': train_last['fp'], 'tn': train_last['tn'], 'fn': train_last['fn']},
            {'split': 'val', 'loss': val_last['loss'], 'accuracy': val_last['accuracy'],
             'precision': val_last['precision'], 'recall': val_last['recall'], 'f1': val_last['f1'], 'mcc': val_last['mcc'],
             'auroc': val_last.get('auroc', float('nan')), 'auprc': val_last.get('auprc', float('nan')),
             'tp': val_last['tp'], 'fp': val_last['fp'], 'tn': val_last['tn'], 'fn': val_last['fn']},
        ])
        summary_df.to_csv(summary_path, index=False)

        # Plot AUROC/AUPRC
        for metric in ['auroc', 'auprc']:
            fig = _plt.figure()
            for split in ['train', 'val']:
                d = df[df['split'] == split]
                _plt.plot(d['epoch'], d[metric], label=split)
            _plt.xlabel('epoch')
            _plt.ylabel(metric)
            _plt.legend()
            _plt.tight_layout()
            fig.savefig(os.path.join(run_dir, f'{metric}_curve.png'), dpi=200)
            _plt.close(fig)

        print('Training summary saved to:', summary_path)
    except Exception as e:
        print('Plot/summary skipped due to error:', e)

    
    # Save the models for predict.py
    model_dir = os.path.join(run_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model_ag, os.path.join(model_dir, 'model_ag.pth'))
    torch.save(model_ab, os.path.join(model_dir, 'model_ab.pth'))
    print("Model training over.")
   
    
def train_model(train_loader, val_loader, model_ag, model_ab, optimizer, criterion, num_epochs, batch_size, threshold, device,
                pin_memory=False, metrics_path=None):
    '''train model'''
    train_losses = []       # Training set loss
    val_losses = []         # validation set loss
    train_accuracies = []   # Training set accuracy
    val_accuracies = []     # Validation set accuracy
    train_x_accuracies = [] # Training set x-axis accuracy
    train_y_accuracies = [] # Training set y-axis accuracy
    val_x_accuracies = []   # Validation set x-axis accuracy
    val_y_accuracies = []   # Validation set y-axis accuracy

    # Mixed precision (GPU-only)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    metrics_file = None
    metrics_writer = None
    if metrics_path:
        metrics_file = open(metrics_path, 'w', newline='')
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([
            'epoch', 'split', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'mcc',
            'auroc', 'auprc',
            'tp', 'fp', 'tn', 'fn'
        ])

    if BinaryAUROC is None or BinaryAveragePrecision is None:
        raise RuntimeError("torchmetrics is required for GPU-only metric computation, but it is not available.")

    use_gpu_metrics = True
    metric_thresholds = 200

    for epoch in range(num_epochs):
        # train
        model_ag.train()
        model_ab.train()

        train_losses_epoch = []
        train_accuracies_epoch = []
        train_x_accuracies_epoch = []
        train_y_accuracies_epoch = []
        
        train_tp = train_fp = train_tn = train_fn = 0
        if use_gpu_metrics:
            train_auroc_metric = BinaryAUROC(thresholds=metric_thresholds).to(device)
            train_auprc_metric = BinaryAveragePrecision(thresholds=metric_thresholds).to(device)
        else:
            train_y_scores_epoch = []
            train_y_true_epoch = []

        for X_ag, X_ab, y_batch in train_loader:
            X_ag = X_ag.to(device, non_blocking=pin_memory)
            X_ab = X_ab.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward + loss
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output_ag = model_ag(X_ag)
                output_ab = model_ab(X_ab)
                result = torch.matmul(output_ag, output_ab.transpose(1, 2))
                result_prob = torch.sigmoid(result)
                loss = criterion(result, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses_epoch.append(loss.item())

            # For AUROC/AUPRC (GPU-only)
            train_auroc_metric.update(result_prob.reshape(-1), y_batch.reshape(-1).int())
            train_auprc_metric.update(result_prob.reshape(-1), y_batch.reshape(-1).int())

            # Calculate the training set accuracy
            train_accuracy = accuracy(y_batch, result_prob, device, threshold)
            train_accuracies_epoch.append(train_accuracy)
            # Calculate x-axis and y-axis accuracy
            train_x_accuracy, train_y_accuracy = calculate_axis_accuracy(y_batch, result_prob,  device)
            train_x_accuracies_epoch.append(train_x_accuracy)
            train_y_accuracies_epoch.append(train_y_accuracy)

            tp, fp, tn, fn = _compute_confusion(y_batch, result_prob, threshold)
            train_tp += tp
            train_fp += fp
            train_tn += tn
            train_fn += fn
            

        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        train_accuracies.append(sum(train_accuracies_epoch) / len(train_accuracies_epoch))
        train_x_accuracies.append(sum(train_x_accuracies_epoch) / len(train_x_accuracies_epoch))
        train_y_accuracies.append(sum(train_y_accuracies_epoch) / len(train_y_accuracies_epoch))        

        # Validate
        model_ag.eval()
        model_ab.eval()

        val_losses_epoch = []
        val_accuracies_epoch = []
        val_x_accuracies_epoch = []
        val_y_accuracies_epoch = []
        if use_gpu_metrics:
            val_auroc_metric = BinaryAUROC(thresholds=metric_thresholds).to(device)
            val_auprc_metric = BinaryAveragePrecision(thresholds=metric_thresholds).to(device)
        else:
            y_scores_epoch = []
            y_true_epoch = []

        val_tp = val_fp = val_tn = val_fn = 0
        with torch.no_grad():
            for X_ag_val, X_ab_val, y_val_batch in val_loader:
                X_ag_val = X_ag_val.to(device, non_blocking=pin_memory)
                X_ab_val = X_ab_val.to(device, non_blocking=pin_memory)
                y_val_batch = y_val_batch.to(device, non_blocking=pin_memory)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    output_ag_val = model_ag(X_ag_val)
                    output_ab_val = model_ab(X_ab_val)
                    result_val = torch.matmul(output_ag_val, output_ab_val.transpose(1, 2))
                    result_val_prob = torch.sigmoid(result_val)

                    # For AUROC/AUPRC (GPU-only)
                    val_auroc_metric.update(result_val_prob.reshape(-1), y_val_batch.reshape(-1).int())
                    val_auprc_metric.update(result_val_prob.reshape(-1), y_val_batch.reshape(-1).int())

                    val_loss = criterion(result_val, y_val_batch)

                val_losses_epoch.append(val_loss.item())
                # Calculate x-axis and y-axis accuracy
                val_x_accuracy, val_y_accuracy = calculate_axis_accuracy(y_val_batch, result_val_prob, device)
                val_x_accuracies_epoch.append(val_x_accuracy)
                val_y_accuracies_epoch.append(val_y_accuracy)

                # Calculate the validation set accuracy
                val_accuracy = accuracy(y_val_batch, result_val_prob, device, threshold)
                val_accuracies_epoch.append(val_accuracy)

                tp, fp, tn, fn = _compute_confusion(y_val_batch, result_val_prob, threshold)
                val_tp += tp
                val_fp += fp
                val_tn += tn
                val_fn += fn

        val_losses.append(sum(val_losses_epoch) / len(val_losses_epoch))
        val_accuracies.append(sum(val_accuracies_epoch) / len(val_accuracies_epoch))
        val_x_accuracies.append(sum(val_x_accuracies_epoch) / len(val_x_accuracies_epoch))
        val_y_accuracies.append(sum(val_y_accuracies_epoch) / len(val_y_accuracies_epoch))

        train_precision, train_recall, train_f1, train_mcc = _metrics_from_confusion(
            train_tp, train_fp, train_tn, train_fn
        )
        val_precision, val_recall, val_f1, val_mcc = _metrics_from_confusion(
            val_tp, val_fp, val_tn, val_fn
        )

        # AUROC/AUPRC (GPU-only)
        try:
            auroc_val = float(val_auroc_metric.compute().item())
        except Exception:
            auroc_val = float('nan')
        try:
            auprc_val = float(val_auprc_metric.compute().item())
        except Exception:
            auprc_val = float('nan')

        try:
            auroc_train = float(train_auroc_metric.compute().item())
        except Exception:
            auroc_train = float('nan')
        try:
            auprc_train = float(train_auprc_metric.compute().item())
        except Exception:
            auprc_train = float('nan')

        if metrics_writer:
            metrics_writer.writerow([
                epoch, 'train',
                round(train_losses[-1], 6), round(train_accuracies[-1], 6),
                round(train_precision, 6), round(train_recall, 6),
                round(train_f1, 6), round(train_mcc, 6),
                round(auroc_train, 6) if not math.isnan(auroc_train) else '',
                round(auprc_train, 6) if not math.isnan(auprc_train) else '',
                train_tp, train_fp, train_tn, train_fn
            ])
            metrics_writer.writerow([
                epoch, 'val',
                round(val_losses[-1], 6), round(val_accuracies[-1], 6),
                round(val_precision, 6), round(val_recall, 6),
                round(val_f1, 6), round(val_mcc, 6),
                round(auroc_val, 6) if not math.isnan(auroc_val) else '',
                round(auprc_val, 6) if not math.isnan(auprc_val) else '',
                val_tp, val_fp, val_tn, val_fn
            ])

        if epoch % 10 == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch, 'loss:',train_losses[-1],'  acc:',train_accuracies[-1])

    if metrics_file:
        metrics_file.close()

        
    # Training summary figure (optional) is handled in an outer block.

    return model_ag, model_ab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EPP model")
    parser.add_argument(
        "--device",
        default=None,
        help="Device setting: auto, cpu, cuda, cuda:0, etc. (default: config)",
    )
    parser.add_argument(
        "--ag-path",
        default=None,
        help="Path to antigen ESM tensor (default: data/traindata_esm_ag.pt)",
    )
    parser.add_argument(
        "--ab-path",
        default=None,
        help="Path to antibody ESM tensor (default: data/traindata_esm_ab.pt)",
    )
    parser.add_argument(
        "--label-path",
        default=None,
        help="Path to label tensor (default: data/label_y.pt)",
    )
    args = parser.parse_args()
    main(
        device_override=args.device,
        ag_path=args.ag_path,
        ab_path=args.ab_path,
        label_path=args.label_path,
    )
