import os
from glob import glob

import _dlib_pybind11
import cv2
import dlib
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError
from easydict import EasyDict

classes = ['angry', 'embarrass', 'fear', 'happy', 'hurt', 'natural', 'sad']
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
EYES = list(range(36, 48))
ALL = list(range(0, 68))

# parser = argparse.ArgumentParser()
# parser.add_argument("-c", "--class_name", required=True)
# parser.add_argument("-t", "--target", required=True)
# parser.add_argument("-g", "--cuda", required=True)
# args = parser.parse_args()

args = EasyDict({
    'class_name': 'angry',
    'target': 'Validation',
    'cuda': 'cuda'
})

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

root = f'/data/korean_emotion/{args.target}/images/{args.class_name}'

class_name = args.class_name
files = glob(os.path.join(root, '*.jpg'))
rect = _dlib_pybind11.rectangle(left=0, right=224, top=0, bottom=224)
import matplotlib.pyplot as plt

for i, f in enumerate(files):
    _, ext = os.path.splitext(f)

    try:
        img = Image.open(f).convert('RGB')

        points = np.matrix([[p.x, p.y] for p in predictor(np.array(img), rect).parts()])
        mouth_mean = np.array(points[MOUTH]).mean(axis=0)
        eyes_mean = np.array(points[EYES]).mean(axis=0)
        print(eyes_mean, mouth_mean)
        img = cv2.circle(np.array(img), (int(eyes_mean[0]), int(eyes_mean[1])), 10, (0, 255, 0), -1)
        img = cv2.circle(np.array(img), (int(mouth_mean[0]), int(mouth_mean[1])), 5, (0, 255, 255), -1)

        plt.imshow(img)
        plt.show()
    except OSError as e:
        pass
    except DecompressionBombError as be:
        pass
    x = input()
    if i == 10:
        break
