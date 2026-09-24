"""Fair, reproducible CPU experiment. Test data are used only after selection."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'data/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import MODELS

ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def evaluate(model, x, y, batch=512):
    model.eval()
    loss, correct, predictions = 0., 0, []
    for start in range(0, len(y), batch):
        target = y[start:start+batch]
        logits = model(x[start:start+batch])
        loss += nn.functional.cross_entropy(logits, target, reduction='sum').item()
        pred = logits.argmax(1)
        correct += (pred == target).sum().item()
        predictions.append(pred)
    return loss/len(y), correct/len(y), torch.cat(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    raw = MNIST(ROOT/'data', train=True, download=True)
    test = MNIST(ROOT/'data', train=False, download=True)
    # Explicit stratified split: 10% of each training class is validation.
    rng = np.random.default_rng(args.seed)
    train_ids, val_ids = [], []
    for digit in range(10):
        ids = rng.permutation(np.flatnonzero(raw.targets.numpy() == digit))
        n = round(len(ids)*.1)
        val_ids.extend(ids[:n]); train_ids.extend(ids[n:])
    train_ids, val_ids = np.array(train_ids), np.array(val_ids)
    np.savez(out/'split_indices.npz', train=train_ids, validation=val_ids)
    # Only scale uint8 pixels to [0,1]; identical normalization at inference.
    x = raw.data.unsqueeze(1).float().div(255)
    tx, ty = x[train_ids], raw.targets[train_ids]
    vx, vy = x[val_ids], raw.targets[val_ids]
    test_x = test.data.unsqueeze(1).float().div(255)
    config = dict(seed=args.seed, epochs=args.epochs, batch_size=128, learning_rate=.001,
                  optimizer='Adam', normalization='uint8 / 255.0', augmentation=False,
                  train_count=len(ty), validation_count=len(vy), test_count=len(test),
                  checkpoint_selection='lowest validation cross entropy',
                  split_sha256=hashlib.sha256(train_ids.tobytes()+val_ids.tobytes()).hexdigest(),
                  torch_version=str(torch.__version__), device='cpu', threads=args.threads)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    histories, results = {}, {}
    for name, cls in MODELS.items():
        seed_all(args.seed)
        model = cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        history, best, best_state, best_epoch = [], float('inf'), None, None
        started = time.perf_counter()
        for epoch in range(1, args.epochs+1):
            model.train()
            # Independent generator ensures exactly the same batches in both runs.
            order = torch.randperm(len(ty), generator=torch.Generator().manual_seed(args.seed+epoch))
            for ids in order.split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(tx[ids]), ty[ids])
                loss.backward()
                optimizer.step()
            tl, ta, _ = evaluate(model, tx, ty)
            vl, va, _ = evaluate(model, vx, vy)
            history.append(dict(epoch=epoch, train_loss=tl, train_accuracy=ta,
                                validation_loss=vl, validation_accuracy=va))
            if vl < best:
                best, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
            print(f'{name.upper()} {epoch:02d}/{args.epochs} train={ta:.4%} val={va:.4%} val_loss={vl:.4f}', flush=True)
            (out/f'{name}_history.json').write_text(json.dumps(history, indent=2))
        seconds = time.perf_counter()-started
        model.load_state_dict(best_state)
        torch.save(dict(model=name, state_dict=best_state, normalization='divide_by_255',
                        selected_epoch=best_epoch, config=config), out/f'{name}.pt')
        test_loss, test_acc, pred = evaluate(model, test_x, test.targets)
        cm = np.zeros((10,10), dtype=int)
        np.add.at(cm, (test.targets.numpy(), pred.numpy()), 1)
        np.save(out/f'{name}_confusion.npy', cm)
        np.save(out/f'{name}_test_predictions.npy', pred.numpy())
        paired = np.triu(cm+cm.T, k=1)
        a,b = np.unravel_index(paired.argmax(), paired.shape)
        results[name] = dict(test_accuracy=test_acc, test_loss=test_loss,
            selected_epoch=best_epoch, best_validation_loss=best,
            parameters=sum(p.numel() for p in model.parameters()), training_seconds=seconds,
            most_confused_pair=[int(a),int(b)], pair_errors=int(paired[a,b]),
            directional_errors={f'{a}->{b}':int(cm[a,b]),f'{b}->{a}':int(cm[b,a])})
        histories[name] = history
        fig, ax = plt.subplots(figsize=(8,7))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(10):
            for j in range(10):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                        color='white' if cm[i,j]>cm.max()/2 else '#172033')
        ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',
               title=f'{name.upper()} test confusion matrix | {test_acc:.2%}')
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/f'{name}_confusion.png',dpi=160); plt.close(fig)
        print(f'{name.upper()} TEST {test_acc:.4%}, selected epoch {best_epoch}',flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for name, history in histories.items():
        epochs = [h['epoch'] for h in history]
        for ax, metric in zip(axes,['loss','accuracy']):
            color = '#2364d2' if name=='cnn' else '#d97706'
            for split, style in [('train','-'),('validation','--')]:
                ax.plot(epochs,[h[f'{split}_{metric}'] for h in history],style,color=color,label=f'{name.upper()} {split}')
            ax.set(xlabel='Epoch',ylabel=metric.title()); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=170); plt.close(fig)


if __name__ == '__main__':
    main()
