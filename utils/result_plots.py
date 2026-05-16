import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_comparison(test_metrics_adam, test_metrics_adamw, test_metrics_scratch):
    experiments = [
        'Adam\n(fine-tuning)',
        'AdamW\n(fine-tuning)',
        'Adam\n(from scratch)'
    ]

    accuracy_values = [
        test_metrics_adam['accuracy'],
        test_metrics_adamw['accuracy'],
        test_metrics_scratch['accuracy']
    ]

    f1_values = [
        test_metrics_adam['f1'],
        test_metrics_adamw['f1'],
        test_metrics_scratch['f1']
    ]

    x = np.arange(len(experiments))
    width = 0.35

    plt.figure(figsize=(12, 6))

    bars_acc = plt.bar(
        x - width / 2,
        accuracy_values,
        width,
        label='Accuracy',
        color='#3498db'
    )

    bars_f1 = plt.bar(
        x + width / 2,
        f1_values,
        width,
        label='F1-score',
        color='#2ecc71'
    )

    plt.title('Comparison of results (Stanford Dogs, ConvNeXt-Tiny + ScratchCNN)')
    plt.xlabel('Experiment')
    plt.ylabel('Value')
    plt.xticks(x, experiments)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    for bars in [bars_acc, bars_f1]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )

    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
