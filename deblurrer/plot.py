import matplotlib.pyplot as plt
import json
import os
import sys

def load_losses(filename='train_losses.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            losses = json.load(f)
            return losses['G_losses'], losses['D_losses']
    else:
        return [], []

if __name__ == '__main__':
    g_avg = []
    d_avg = []
    if len(sys.argv) < 2:
        print("Usage: python plot.py [train|test]")
        sys.exit(1)
    if sys.argv[1] == 'train':
        G_losses, D_losses = load_losses()
        G_losses = [G_losses[i:i+2388] for i in range(0, len(G_losses), 2388)]
        D_losses = [D_losses[i:i+2388] for i in range(0, len(D_losses), 2388)]
        for i in range(100):
            g_avg.append(sum(G_losses[i]) / len(G_losses[i]))
            d_avg.append(sum(D_losses[i]) / len(D_losses[i]))
    elif sys.argv[1] == 'test':
        g_avg, d_avg = load_losses('test_losses.json')
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Generator Loss', color=color)
    ax1.plot(g_avg, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Discreminator Loss', color=color)
    ax2.plot(d_avg, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    ax1.set_xticks(range(0, 100, 1))
    ax1.set_xticklabels([str(i + 1) for i in range(0, 100, 1)], rotation=-90)
    for i in range(0, 100, 1):
        ax1.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
    plt.show()

    g_avg = [g_avg[i] / (sum(g_avg) / len(g_avg)) for i in range(1, len(g_avg))]
    d_avg = [d_avg[i] / (sum(d_avg) / len(d_avg)) for i in range(1, len(d_avg))]
    min_product = float('inf')
    min_index = -1

    for i in range(len(g_avg)):
        product = g_avg[i] + d_avg[i]
        if product < min_product:
            min_product = product
            min_index = i

    print(min_index)