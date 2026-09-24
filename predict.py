"""Standalone inference: loads saved weights; never trains or loads MNIST."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from models import MODELS

ROOT = Path(__file__).resolve().parent


def preprocess(image, polarity='auto'):
    image = ImageOps.exif_transpose(image)
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, 'white')
        image = Image.alpha_composite(background, image)
    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray, dtype=np.float32)
    border = np.concatenate([arr[0],arr[-1],arr[:,0],arr[:,-1]])
    invert = polarity == 'dark' or (polarity == 'auto' and np.median(border)>127)
    ink = ImageOps.invert(gray) if invert else gray.copy()
    ink = ImageOps.autocontrast(ink)
    arr = np.asarray(ink)
    yy, xx = np.nonzero(arr > 40)
    if len(xx) < 3:
        raise ValueError('No digit detected. Draw a visible digit first.')
    crop = ink.crop((int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1))
    w,h = crop.size
    resized = crop.resize((max(1,round(w*20/max(w,h))),max(1,round(h*20/max(w,h)))),Image.Resampling.LANCZOS)
    canvas = Image.new('L',(28,28),0)
    canvas.paste(resized,((28-resized.width)//2,(28-resized.height)//2))
    a = np.asarray(canvas,dtype=np.float32)
    yy,xx = np.indices(a.shape)
    dx,dy = round(13.5-float((xx*a).sum()/a.sum())), round(13.5-float((yy*a).sum()/a.sum()))
    centered = Image.new('L',(28,28),0)
    centered.paste(canvas,(dx,dy))
    tensor = torch.from_numpy(np.array(centered,dtype=np.float32)/255.).unsqueeze(0).unsqueeze(0)
    return tensor, {'grayscale':gray,'inverted':ink,'resized':resized,'normalized':centered}


def load_model(path=ROOT/'artifacts/cnn.pt'):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['normalization'] != 'divide_by_255':
        raise ValueError('Unsupported normalization')
    model = MODELS[checkpoint['model']]()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


@torch.inference_mode()
def predict(model, image, polarity='auto'):
    tensor, stages = preprocess(image,polarity)
    probs = model(tensor).softmax(1)[0]
    scores, digits = probs.topk(3)
    return dict(predicted_digit=int(digits[0]),confidence=float(scores[0]),
                top3=[dict(digit=int(d),probability=float(p)) for d,p in zip(digits,scores)]), stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/cnn.pt')
    parser.add_argument('--polarity',choices=['auto','dark','light'],default='auto',help='Ink color')
    parser.add_argument('--stages',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with Image.open(args.image) as image:
        result, stages = predict(load_model(args.model),image,args.polarity)
    if args.stages:
        args.stages.mkdir(parents=True,exist_ok=True)
        for name, image in stages.items():
            image.save(args.stages/f'{name}.png')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
