import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from models.ST_Former import GenerateModel
from dataloader.dataset_DFEW import test_data_loader

# Emotion class labels corresponding to outputs 0-6 in DFEW
EMOTION_NAMES = ['Happiness', 'Sadness', 'Neutral', 'Anger', 'Surprise', 'Disgust', 'Fear']

class RecorderMeter(object):
    """Placeholder class required for unpickling checkpoints trained via main_DFEW.py"""
    def __init__(self, total_epoch):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Former-DFER across all 5 DFEW folds")
    parser.add_argument('--ckpt_set1', type=str, required=True, help='Path to set1 best checkpoint')
    parser.add_argument('--ckpt_set2', type=str, required=True, help='Path to set2 best checkpoint')
    parser.add_argument('--ckpt_set3', type=str, required=True, help='Path to set3 best checkpoint')
    parser.add_argument('--ckpt_set4', type=str, required=True, help='Path to set4 best checkpoint')
    parser.add_argument('--ckpt_set5', type=str, required=True, help='Path to set5 best checkpoint')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='Batch size for evaluation')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Data loading workers')
    return parser.parse_args()

def eval_fold(model, fold_num, ckpt_path, batch_size, workers):
    print(f"Evaluating Fold {fold_num} with checkpoint: {ckpt_path}...")
    
    test_data = test_data_loader(data_set=fold_num)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    checkpoint = torch.load(ckpt_path, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    fold_preds = []
    fold_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            fold_preds.extend(preds.cpu().numpy())
            fold_targets.extend(targets.numpy())

    return np.array(fold_preds), np.array(fold_targets)

def main():
    args = parse_args()
    ckpts = [
        args.ckpt_set1,
        args.ckpt_set2,
        args.ckpt_set3,
        args.ckpt_set4,
        args.ckpt_set5
    ]

    # Initialize model
    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda()

    all_preds = []
    all_targets = []

    # Iterate through all 5 folds
    for fold_idx in range(1, 6):
        preds, targets = eval_fold(model, fold_idx, ckpts[fold_idx - 1], args.batch_size, args.workers)
        all_preds.extend(preds)
        all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate overall confusion matrix across all 5 folds
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(EMOTION_NAMES))))

    # Per-class accuracy (Recall per emotion)
    class_accs = (cm.diagonal() / cm.sum(axis=1)) * 100.0

    # WAR: Overall accuracy across all samples
    war = (np.sum(cm.diagonal()) / np.sum(cm)) * 100.0

    # UAR: Mean of individual class accuracies
    uar = np.mean(class_accs)

    print("\n" + "=" * 60)
    print("               DFEW 5-FOLD EVALUATION RESULTS               ")
    print("=" * 60)
    print(f"{'Emotion':<12} | {'Your Accuracy (%)':<18} | {'Paper Result (%)':<18}")
    print("-" * 55)
    
    paper_accs = [84.05, 62.57, 67.52, 70.03, 56.43, 3.45, 31.78]
    for name, acc, paper_acc in zip(EMOTION_NAMES, class_accs, paper_accs):
        print(f"{name:<12} | {acc:<18.2f} | {paper_acc:<18.2f}")
        
    print("-" * 55)
    print(f"{'UAR':<12} | {uar:<18.2f} | {53.69:<18.2f}")
    print(f"{'WAR (Acc)':<12} | {war:<18.2f} | {65.70:<18.2f}")
    print("=" * 60)

if __name__ == '__main__':
    main()