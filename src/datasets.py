from __future__ import annotations

import torchvision.transforms.transforms as trf
from pathlib import Path
import torch as T
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import snntorch.spikegen as sg
from functools import partial
import typing as t


class DatasetPreprocessed(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.files_X = sorted(self.folder.glob('X_*.pt'))
        self.files_y = sorted(self.folder.glob('y_*.pt'))
        self.num_pairs = len(self.files_X)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        X = T.load(self.files_X[idx])
        y = T.load(self.files_y[idx])
        return X, y


def get_default_mnist_transforms() -> trf.Compose:
    return trf.Compose([
        trf.Resize((28, 28)),
        trf.Grayscale(),
        trf.ToTensor(),
        trf.Lambda(lambda x: T.flatten(x)),
        trf.Normalize((0,), (1,))
    ])


def get_mnist_processed(
    train: bool = True,
    cache_dir: Path = Path('./data'),
    transforms: trf.Compose | None = None,
) -> tuple[T.Tensor, T.Tensor]:
    cache_dir_mnist_processed = cache_dir / 'mnist_processed'
    prefix = 'train' if train else 'test'
    path_X = cache_dir_mnist_processed / f'processed_X_{prefix}.pt'
    path_y = cache_dir_mnist_processed / f'processed_y_{prefix}.pt'

    if path_X.exists() and path_y.exists():
        X = T.load(path_X)
        y = T.load(path_y)
    else:
        if transforms is None:
            transforms = get_default_mnist_transforms()
        dataset = tv.datasets.MNIST(root=str(cache_dir), train=train, download=True, transform=transforms)
        train_loader = DataLoader(dataset, batch_size=len(dataset.targets), shuffle=True, drop_last=True)

        X, y = next(iter(train_loader))
        X = X.reshape((X.size(0), -1))

        y, y_temp = T.zeros((y.size(0), 10)), y
        y[T.arange(y_temp.size(0)), y_temp] = 1

        cache_dir_mnist_processed.mkdir(parents=True)
        T.save(X, path_X)
        T.save(y, path_y)

    return X, y


def _get_mnist_snn_encoding(
    encoding_type_label: str,
    encoding_settings_label: str,
    enc_function: t.Callable,
    batch_size: int,
    train: bool,
    cache_dir: Path = Path('./data/'),
    transforms: trf.Compose | None = None,
) -> Dataset:
    cache_dir_mnist_snn = cache_dir / encoding_type_label / encoding_settings_label
    prefix = 'train' if train else 'test'

    path_X_template = f'X_{prefix}__batch={{i}}.pt'
    path_y_template = f'y_{prefix}__batch={{i}}.pt'

    if not cache_dir_mnist_snn.exists():
        X, y = get_mnist_processed(train=train, cache_dir=cache_dir, transforms=transforms)

        cache_dir_mnist_snn.mkdir(parents=True)
        for i in range(0, X.size(0), batch_size):
            idx_start = i*batch_size
            idx_end = (i+1)*batch_size
            X_i = enc_function(X[idx_start:idx_end, :])
            y_i = y[idx_start:idx_end, :]

            T.save(X_i, cache_dir_mnist_snn / path_X_template.format(i=i))
            T.save(y_i, cache_dir_mnist_snn / path_y_template.format(i=i))

    return DatasetPreprocessed(cache_dir_mnist_snn)


def get_mnist_snn_encoding__rate(
    batch_size: int,
    num_steps: int,
    train: bool,
) -> Dataset:
    return DataLoader(
        _get_mnist_snn_encoding(
            encoding_type_label='mnist_rate_encoded',
            encoding_settings_label=f'batch_size={batch_size}__num_steps={num_steps}',
            enc_function=partial(sg.rate, num_steps=num_steps),  # type: ignore
            batch_size=batch_size,
            train=train,
        ),
        pin_memory=True,
        # TODO: Add additional params as needed.
    )


def get_mnist_snn_encoding__latency(
    batch_size: int,
    num_steps: int,
    tau: float,
    threshold: float,
    train: bool,
) -> Dataset:
    return DataLoader(
        _get_mnist_snn_encoding(
            encoding_type_label='mnist_latency',
            encoding_settings_label=f'batch_size={batch_size}__num_steps={num_steps}__tau={tau:.5f}__threshold={threshold:.5f}',
            enc_function=partial(sg.latency, num_steps=num_steps, tau=tau, threshold=threshold),  # type: ignore
            batch_size=batch_size,
            train=train,
        ),
        pin_memory=True,
        # TODO: Add additional params as needed.
    )


if __name__ == '__main__':
    import time
    start_time = time.time()
    dl = get_mnist_snn_encoding__rate(256, 128)
    X, y = next(iter(dl))
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds')
    print(f"Size of rate encoded data: {X.size()}")

    start_time = time.time()
    dl = get_mnist_snn_encoding__latency(256, 128, tau=5, threshold=0.01)
    X, y = next(iter(dl))
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds')
    print(f"Size of rate encoded data: {X.size()}")
