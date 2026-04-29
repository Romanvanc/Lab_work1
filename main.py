import os
import torch
import torch.nn as nn
import random
import numpy as np

from models import get_convnext_tiny
from utils import load_datasets, create_loaders, train_model, evaluate, plot_comparison

def set_seed(seed=497328):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    print("\nLoading Stanford Dogs dataset...")
    train_dataset, val_dataset, test_dataset, num_classes, _ = load_datasets()
    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)
    
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 10
    
    results = {}
    
    # Experiment 1: Finetune + Adam
    print("\n" + "="*60)
    print("EXPERIMENT 1: ConvNeXt-Tiny + Adam (Finetune)")
    model1 = get_convnext_tiny(pretrained=True, num_classes=num_classes)
    for param in model1.parameters():
        param.requires_grad = False
    in_features = model1.classifier[2].in_features
    model1.classifier[2] = nn.Linear(in_features, num_classes)
    for param in model1.classifier[2].parameters():
        param.requires_grad = True
    model1 = model1.to(device)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
    model1, history1 = train_model(model1, train_loader, val_loader, optimizer1,
                                    criterion, EPOCHS, device, "convnext_adam")
    test_metrics1 = evaluate(model1, test_loader, criterion, device)
    results['Adam (Finetune)'] = test_metrics1
    
    # Experiment 2: Finetune + AdamW (variant requirement)
    print("\n" + "="*60)
    print("EXPERIMENT 2: ConvNeXt-Tiny + AdamW (Finetune)")
    model2 = get_convnext_tiny(pretrained=True, num_classes=num_classes)
    for param in model2.parameters():
        param.requires_grad = False
    model2.classifier[2] = nn.Linear(in_features, num_classes)
    for param in model2.classifier[2].parameters():
        param.requires_grad = True
    model2 = model2.to(device)
    
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
    model2, history2 = train_model(model2, train_loader, val_loader, optimizer2,
                                    criterion, EPOCHS, device, "convnext_adamw")
    test_metrics2 = evaluate(model2, test_loader, criterion, device)
    results['AdamW (Finetune)'] = test_metrics2
    
    # Experiment 3: From scratch + Adam
    print("\n" + "="*60)
    print("EXPERIMENT 3: ConvNeXt-Tiny + Adam (From Scratch)")
    model3 = get_convnext_tiny(pretrained=False, num_classes=num_classes)
    model3 = model3.to(device)
    
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3, weight_decay=1e-4)
    model3, history3 = train_model(model3, train_loader, val_loader, optimizer3,
                                    criterion, EPOCHS, device, "convnext_scratch")
    test_metrics3 = evaluate(model3, test_loader, criterion, device)
    results['Adam (Scratch)'] = test_metrics3
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"{'Experiment':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 68)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Plot comparison
    plot_comparison(results)
    print("\nPlot saved to results/comparison_results.png")

if __name__ == "__main__":
    main()
