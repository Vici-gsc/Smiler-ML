import os
import random
from glob import glob
from math import floor

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from src.data import register_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


@register_dataset
class CelebA(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        self.std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        self.resize = (int(floor(224 / 0.95)), int(floor(224 / 0.95)))
        self.x = list()

        self.x = glob(os.path.join(root, 'img_align_celeba', '*'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        upside_down = random.randrange(0, 2)

        transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(upside_down),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        image = transform(image)

        return image, upside_down


if __name__ == '__main__':
    ds = CelebA('/data/celeba', 'a')
    img, label = next(iter(ds))
    print(img.shape)
    print(label)
