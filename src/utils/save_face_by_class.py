import argparse
import os
from glob import glob

import dlib
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError
from facenet_pytorch.models.mtcnn import MTCNN
from tqdm import tqdm

classes = ['angry', 'embarrass', 'fear', 'happy', 'hurt', 'natural', 'sad']

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--class_name", required=True)
parser.add_argument("-t", "--target", required=True)
parser.add_argument("-g", "--cuda", required=True)
args = parser.parse_args()

mtcnn = MTCNN(image_size=224, post_process=False, device=f'cuda:{args.cuda}')

detector = dlib.get_frontal_face_detector()
root = f'/data/korean_emotion/{args.target}/images_large/{args.class_name}'
face_root = f'/data/korean_emotion/{args.target}/images'

if not os.path.exists(face_root):
    os.mkdir(face_root)

class_name = args.class_name
files = glob(os.path.join(root, '*.jpg'))

save_path = os.path.join(face_root, class_name)

if not os.path.exists(os.path.join(face_root, class_name)):
    os.mkdir(save_path)
for i, f in enumerate(tqdm(files)):
    try:
        img = Image.open(f)

        if len(detector(np.asarray(img), 1)) < 1:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        _, ext = os.path.splitext(f)
        face, prob = mtcnn(img, save_path=os.path.join(save_path, f"{i}{ext}"), return_prob=True)
    except OSError as e:
        pass
    except DecompressionBombError as be:
        pass
