"""Reload saved weights and test non-MNIST handwriting images."""
import json
import os
from pathlib import Path
import torch
from PIL import Image
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'data/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from predict import ROOT, load_model, predict


def main():
    torch.set_num_threads(2)
    folder = ROOT/'custom_images'
    manifest = json.loads((folder/'manifest.json').read_text())
    samples = manifest['samples']
    if not samples:
        raise ValueError('The manifest must contain labeled images.')
    listed = {sample['file'] for sample in samples}
    actual = {p.name for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}}
    if listed != actual or len(listed) != len(samples):
        raise ValueError('List every image exactly once in manifest.json, with its true_digit.')
    model = load_model()
    results = []
    fig, axes = plt.subplots(len(samples), 4, figsize=(10, 2.4 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        with Image.open(folder/sample['file']) as image:
            original = image.copy()
        result, stages = predict(model, original)
        label = sample['true_digit']
        results.append(dict(file=sample['file'], true_digit=label, **result))
        print(f"{sample['file']}: True: {label} | Predicted: {result['predicted_digit']}")
        images = [('Original', original), ('Grayscale', stages['grayscale']),
                  ('Invert', stages['inverted']), ('Resize / center / 255', stages['normalized'])]
        for ax, (title, image) in zip(axes[row], images):
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        axes[row, 3].set_title(f"True {label} / predicted {result['predicted_digit']}", fontsize=10)
    out = ROOT/'artifacts'
    out.mkdir(exist_ok=True)
    fig.suptitle(f'{len(samples)} handwritten images: preprocessing')
    fig.tight_layout()
    fig.savefig(out/'custom_preprocessing.png', dpi=150)
    plt.close(fig)
    report = dict(source=manifest['source'],
                  selection=manifest['selection'], note=manifest['note'],
                  count=len(samples), correct=sum(r['true_digit']==r['predicted_digit'] for r in results),
                  samples=results)
    (out/'custom_results.json').write_text(json.dumps(report, indent=2))
    print(f"Correct: {report['correct']}/{len(samples)}")


if __name__ == '__main__':
    main()
