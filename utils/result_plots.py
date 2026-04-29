import matplotlib.pyplot as plt

def plot_comparison(results):
    experiments = list(results.keys())
    accuracies = [results['accuracy'] for results in results.values()]
    f1_scores = [results['f1'] for results in results.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(experiments))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], accuracies, width, 
                   label='Accuracy', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], f1_scores, width,
                   label='F1-score', color='#2ecc71')
    
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Results (Stanford Dogs, ConvNeXt-Tiny)')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=150)
    plt.show()

def plot_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_history.png', dpi=150)
    plt.show()
