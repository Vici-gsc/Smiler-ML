import os
from glob import glob

from PIL import Image
from PIL.Image import DecompressionBombError
from tqdm import tqdm

root = '/data/korean_emotion/Validation/'
classes = glob('/data/korean_emotion/Validation/images/*')
small_root = '/data/korean_emotion/Validation/images_small'

if not os.path.exists(small_root):
    os.mkdir(small_root)

for c in classes:
    class_name = os.path.basename(c)
    files = glob(os.path.join(c, '*'))

    if not os.path.exists(os.path.join(small_root, class_name)):
        os.mkdir(os.path.join(small_root, class_name))
    for i, f in tqdm(enumerate(files), total=len(files)):
        try:
            img = Image.open(f)
            img_resize = img.resize((int(img.width / 2), int(img.height / 2)))
            title, ext = os.path.splitext(f)
            save_path = os.path.join(small_root, class_name, title + '_half' + ext)
            img_resize.save(save_path)
        except OSError as e:
            pass
        except DecompressionBombError as be:
            pass
