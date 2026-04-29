import torch
import torch.nn as nn
import random
import numpy as np

from data.download_dataset import download_stanford_dogs
from utils.dataset import load_datasets, create_loaders
from utils.train import train_model, evaluate
from utils.visualization import plot_comparison

from models import convnext_adam, convnext_adamw, convnext_scratch

def set_seed(seed=497328):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Uncomment for first run
    # download_stanford_dogs()
    
    train_dataset, val_dataset, test_dataset, num_classes, _ = load_datasets()
    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)
    
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 10
    
    results = {}
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: ConvNeXt-Tiny + Adam (finetune)")
    model1 = convnext_adam.get_model(num_classes).to(device)
    optimizer1 = convnext_adam.get_optimizer(model1)
    history1, best_acc1 = train_model(model1, train_loader, val_loader, optimizer1, criterion, EPOCHS, device, "convnext_adam")
    test_metrics1 = evaluate(model1, test_loader, criterion, device)
    results['Adam'] = test_metrics1
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: ConvNeXt-Tiny + AdamW (finetune)")
    model2 = convnext_adamw.get_model(num_classes).to(device)
    optimizer2 = convnext_adamw.get_optimizer(model2)
    history2, best_acc2 = train_model(model2, train_loader, val_loader, optimizer2, criterion, EPOCHS, device, "convnext_adamw")
    test_metrics2 = evaluate(model2, test_loader, criterion, device)
    results['AdamW'] = test_metrics2
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: ConvNeXt-Tiny + Adam (from scratch)")
    model3 = convnext_scratch.get_model(num_classes).to(device)
    optimizer3 = convnext_scratch.get_optimizer(model3)
    history3, best_acc3 = train_model(model3, train_loader, val_loader, optimizer3, criterion, EPOCHS, device, "convnext_scratch")
    test_metrics3 = evaluate(model3, test_loader, criterion, device)
    results['Scratch'] = test_metrics3
    
    print("\n" + "="*60)
    print("TEST RESULTS:")
    print("-"*40)
    for name, metrics in results.items():
        print(f"{name:10} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    
    plot_comparison()
    print("\nExperiments completed. Plot saved to results/")

if __name__ == "__main__":
    main()
