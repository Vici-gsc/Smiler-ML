from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.samplers import DistributedSampler

from src.data.transform import TrainTransform, ValTransform

_dataset_dict = {}


def register_dataset(fn):
    dataset_name = fn.__name__
    if dataset_name not in _dataset_dict:
        _dataset_dict[fn.__name__] = fn
    else:
        raise ValueError(f"{dataset_name} already exists in dataset_dict")

    return fn


def get_dataloader(cfg):
    dataset_class = _dataset_dict[cfg.dataset.name]

    ds_train = dataset_class(root=cfg.dataset.root, mode='train', transform=TrainTransform())
    ds_valid = dataset_class(root=cfg.dataset.root, mode='valid', transform=ValTransform())

    if cfg.distributed:
        train_sampler = DistributedSampler(ds_train, shuffle=True)
        val_sampler = DistributedSampler(ds_valid, shuffle=False)
    else:
        train_sampler = RandomSampler(ds_train)
        val_sampler = SequentialSampler(ds_valid)

    dl_train = DataLoader(ds_train, batch_size=cfg.train.batch_size, sampler=train_sampler,
                          num_workers=cfg.train.num_workers, collate_fn=None, pin_memory=True)

    dl_valid = DataLoader(ds_valid, batch_size=cfg.train.batch_size, sampler=val_sampler,
                          num_workers=cfg.train.num_workers, collate_fn=None, pin_memory=False)

    return dl_train, dl_valid
