"""Plot the first convolution layer of the saved CNN."""
import os
from predict import ROOT, load_model
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    model = load_model()
    filters = model.features[0].weight.detach().numpy()[:, 0]
    limit = abs(filters).max()
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for number, (ax, kernel) in enumerate(zip(axes.flat, filters), start=1):
        ax.imshow(kernel, cmap='coolwarm', vmin=-limit, vmax=limit)
        ax.set_title(str(number), fontsize=9)
        ax.axis('off')
    fig.suptitle('First convolution layer: 16 learned 3 x 3 filters')
    fig.tight_layout()
    output = ROOT/'artifacts/filters.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
