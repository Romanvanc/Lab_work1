import matplotlib.pyplot as plt

def plot_comparison():
    experiments = ['Adam (finetune)', 'AdamW (finetune)', 'Adam (scratch)']
    accuracies = [0.9459, 0.9427, 0.0460]
    f1_scores = [0.9453, 0.9423, 0.0300]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(experiments))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='F1-score', color='#2ecc71')
    
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
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=150)
    plt.show()
