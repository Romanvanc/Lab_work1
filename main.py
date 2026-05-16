import os
import random

import numpy as np
import torch
import torch.nn as nn

from models import get_convnext_tiny
from models.scratch import get_scratch_cnn
from utils import load_datasets, create_loaders, train_model, evaluate, plot_comparison
from utils.result_plots import plot_training_history


def set_seed(seed=497328):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = 'classifier.2' in name
    return model


def print_test_metrics(title, metrics):
    print(f"\nTest results ({title}):")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-score:  {metrics['f1']:.4f}")


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)

    train_dataset, val_dataset, test_dataset, num_classes, _ = load_datasets()
    train_loader, val_loader, test_loader = create_loaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    criterion = nn.CrossEntropyLoss()
    results = {}

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: ConvNeXt-Tiny + Adam (Fine-tuning)")
    model_adam = get_convnext_tiny(pretrained=True, num_classes=num_classes)
    model_adam = freeze_backbone(model_adam).to(device)

    optimizer_adam = torch.optim.Adam(
        model_adam.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    model_adam, history_adam = train_model(
        model_adam,
        train_loader,
        val_loader,
        optimizer_adam,
        criterion,
        epochs=10,
        device=device,
        model_name="convnext_adam"
    )

    test_metrics_adam = evaluate(model_adam, test_loader, criterion, device)
    print_test_metrics("ConvNeXt-Tiny + Adam", test_metrics_adam)
    results["Adam (Fine-tuning)"] = test_metrics_adam
    plot_training_history(history_adam, "adam")

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: ConvNeXt-Tiny + AdamW (Fine-tuning)")
    model_adamw = get_convnext_tiny(pretrained=True, num_classes=num_classes)
    model_adamw = freeze_backbone(model_adamw).to(device)

    optimizer_adamw = torch.optim.AdamW(
        model_adamw.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )

    model_adamw, history_adamw = train_model(
        model_adamw,
        train_loader,
        val_loader,
        optimizer_adamw,
        criterion,
        epochs=10,
        device=device,
        model_name="convnext_adamw"
    )

    test_metrics_adamw = evaluate(model_adamw, test_loader, criterion, device)
    print_test_metrics("ConvNeXt-Tiny + AdamW", test_metrics_adamw)
    results["AdamW (Fine-tuning)"] = test_metrics_adamw
    plot_training_history(history_adamw, "adamw")

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: ScratchCNN + Adam (From Scratch)")
    model_scratch = get_scratch_cnn(num_classes=num_classes).to(device)

    trainable_params = sum(p.numel() for p in model_scratch.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_scratch.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    optimizer_scratch = torch.optim.Adam(
        model_scratch.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    model_scratch, history_scratch = train_model(
        model_scratch,
        train_loader,
        val_loader,
        optimizer_scratch,
        criterion,
        epochs=40,
        device=device,
        model_name="scratch_cnn"
    )

    test_metrics_scratch = evaluate(model_scratch, test_loader, criterion, device)
    print_test_metrics("custom ScratchCNN from scratch + Adam", test_metrics_scratch)
    results["ScratchCNN + Adam"] = test_metrics_scratch
    plot_training_history(history_scratch, "scratch")

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"{'Experiment':<24} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 76)

    for name, metrics in results.items():
        print(
            f"{name:<24} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f}"
        )

    plot_comparison(results)
    print("\nPlots saved to results/")


if __name__ == "__main__":
    main()
