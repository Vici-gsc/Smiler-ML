import os
from glob import glob

import dlib
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError
from facenet_pytorch.models.mtcnn import MTCNN

mtcnn = MTCNN(image_size=224, post_process=False, device='cuda')

detector = dlib.get_frontal_face_detector()
root = '/data/korean_emotion/Validation/'
classes = glob('/data/korean_emotion/Validation/images_large/*')
face_root = '/data/korean_emotion/Validation/images'

if not os.path.exists(face_root):
    os.mkdir(face_root)

for c in classes:
    print(c)
    class_name = os.path.basename(c)
    files = glob(os.path.join(c, '*'))

    save_path = os.path.join(face_root, class_name)

    if not os.path.exists(os.path.join(face_root, class_name)):
        os.mkdir(save_path)
    for i, f in enumerate(files):
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
