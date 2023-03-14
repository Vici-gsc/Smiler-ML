import os.path
from glob import glob

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from src.data import register_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


@register_dataset
class KoreanFaceEmotion(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        self.classes = {c: i for i, c in enumerate(['natural', 'angry', 'embarrass', 'fear', 'happy', 'hurt', 'sad'])}
        self.x = list()
        self.y = list()

        if mode == 'train':
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        for key, value in self.classes.items():
            image_paths = glob(os.path.join(self.root, key, '*'))
            for path in image_paths:
                self.x.append(path)
                self.y.append(value)

        assert len(self.x) == len(self.y), f'The sizes of x and y are different. {len(self.x)}!={len(self.y)}'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.ToTensor()
    ds = KoreanFaceEmotion('/data/korean_emotion', 'valid', transform)
    img, label = next(iter(ds))
    print(img.shape)
    print(label)
