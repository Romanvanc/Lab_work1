import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(history, model_name, save_path='results'):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}_history.png', dpi=150)
    plt.close()

def plot_confusion_matrix(model, loader, device, num_classes, class_names, model_name, save_path='results'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=False, fmt='.1f', cmap='Blues', 
                xticklabels=class_names if len(class_names) <= 30 else [],
                yticklabels=class_names if len(class_names) <= 30 else [])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}_confusion_matrix.png', dpi=150)
    plt.close()
    
    # Вывод общей точности
    acc = np.trace(cm) / np.sum(cm)
    print(f"{model_name} - Confusion matrix saved, accuracy: {acc:.4f}")

def plot_comparison(results, save_path='results'):
    experiments = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in experiments]
    f1_scores = [results[name]['f1'] for name in experiments]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(experiments))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='F1-score', color='#2ecc71')
    
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Results')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comparison_results.png', dpi=150)
    plt.close()
