import argparse
import torch
import torch.nn as nn
from models.ST_Former import GenerateModel
from dataloader.dataset_DFEW import test_data_loader

# Define RecorderMeter so PyTorch unpickler finds it in __main__
class RecorderMeter(object):
    """Placeholder to allow unpickling checkpoints trained in main_DFEW.py"""
    def __init__(self, total_epoch):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Former-DFER model")
    parser.add_argument('--weights', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_set', type=int, default=1, help='DFEW fold set number (1-5)')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    return parser.parse_args()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    args = parse_args()
    
    print(f"Loading test dataset for DFEW Set {args.data_set}...")
    test_data = test_data_loader(data_set=args.data_set)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print("Building model architecture...")
    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda()

    print(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, weights_only=False)
    
    # Handle state dictionary loading
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    top1 = 0.0
    total_samples = 0

    print("Running evaluation...")
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            acc1, _ = accuracy(output, target, topk=(1, 5))

            batch_size = images.size(0)
            top1 += acc1[0].item() * batch_size
            total_samples += batch_size

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"Batch [{i + 1}/{len(test_loader)}] | Current Accuracy: {top1 / total_samples:.3f}%")

    final_accuracy = top1 / total_samples
    print("\n" + "=" * 40)
    print(f"Final Test Accuracy (Set {args.data_set}): {final_accuracy:.3f}%")
    print("=" * 40)

if __name__ == '__main__':
    main()