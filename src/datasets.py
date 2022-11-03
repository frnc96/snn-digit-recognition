from __future__ import annotations

import typing as t
from functools import partial
from pathlib import Path

import snntorch.spikegen as sg
import torch as T
import torchvision as tv
import torchvision.transforms.transforms as trf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


class DatasetOnDisk(Dataset):
    """A dataset that is preprocessed and saved to disk, for data that doesn't fit into memory."""

    def __init__(self, folder: Path):
        # Set up references to the files in the dataset.
        self.folder = Path(folder)
        self.files_X = sorted(self.folder.glob('X_*.pt'))
        self.files_y = sorted(self.folder.glob('y_*.pt'))
        self.num_pairs = len(self.files_X)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        """Return a data pair."""
        X = T.load(self.files_X[idx])
        y = T.load(self.files_y[idx])

        return X, y


class DatasetInMemory(DatasetOnDisk):
    """A dataset that loads a DatasetOnDisk into memory."""

    def __init__(self, folder: Path, device: T.device):
        super().__init__(folder)

        X = T.load(self.files_X[0])
        for file in tqdm(self.files_X[1:]):
            X = T.cat((X, T.load(file)), dim=0)
        X = X.to(device)
        self.X = X

        y = T.load(self.files_y[0])
        for path in tqdm(self.files_y[1:]):
            y = T.cat((y, T.load(path)))
        y = y.to(device)
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        """Return a data pair."""
        return self.X[idx, :], self.y[idx, :]


def get_default_mnist_transforms() -> trf.Compose:
    """A basic set of transforms for MNIST.

    The images are resized to 28x28, converted to grayscale, normalized to the range [0, 1],
    then flattened to a vector og 28x28=784 elements.

    Returns:
        A torchvision.transforms.Compose object.
    """
    return trf.Compose([
        trf.ToTensor(),
        trf.Lambda(lambda x: T.flatten(x)),
    ])


def get_mnist_processed(
    train: bool = True,
    cache_dir: Path = Path('./data'),
    transforms: trf.Compose | None = None,
) -> tuple[T.Tensor, T.Tensor]:
    """Get the MNIST dataset as raw vectors.

    - The X data is a tensor of shape (N, 784), where N is the number of images.
    - The y data is a tensor of shape (N, 10), where each row is a one-hot vector.

    The results are preprocessed and saved to disk for later use.

    Args:
        train (bool): Whether to load the training or test set.
        cache_dir (Path): The directory to save the preprocessed data to.
        transforms (torchvision.transforms.Compose): The transforms to apply to the data.
    Returns:
        A tuple of tensors (X, y) containing the data and labels.
    """

    # Sets up the MNIST dataset folder for the preprocessed data.
    cache_dir_mnist_processed = cache_dir / 'mnist_processed'

    # Define the filenames for the preprocessed data.
    prefix = 'train' if train else 'test'
    path_X = cache_dir_mnist_processed / f'processed_X_{prefix}.pt'
    path_y = cache_dir_mnist_processed / f'processed_y_{prefix}.pt'

    if path_X.exists() and path_y.exists():
        # Assume the data has been processed if the files exist.
        X = T.load(path_X)
        y = T.load(path_y)
    else:
        if transforms is None:
            transforms = get_default_mnist_transforms()

        # Load all the data from the MNIST dataset.
        dataset = tv.datasets.MNIST(root=str(cache_dir), train=train, download=True, transform=transforms)
        data_loader = DataLoader(dataset, batch_size=len(dataset.targets), shuffle=True, drop_last=True)

        X, y = next(iter(data_loader))

        # Make sure x is the correct size.
        X = X.reshape((X.size(0), -1))

        # Convert y to a one-hot vector, that is (N, 1) -> (N, 10).
        y, y_temp = T.zeros((y.size(0), 10)), y
        y[T.arange(y_temp.size(0)), y_temp] = 1

        # Save the data to disk for later use.
        cache_dir_mnist_processed.mkdir(parents=True, exist_ok=True)
        T.save(X, path_X)
        T.save(y, path_y)

    return X, y


def _get_mnist_dataset_spike_encoded(
    encoding_type_label: str,
    encoding_settings_label: str,
    enc_function: t.Callable,
    batch_size: int,
    train: bool,
    cache_dir: Path,
    transforms: trf.Compose | None,
    show_progress: bool,
    in_memory: bool,
    device: T.device,
) -> Dataset:
    """A base function to easily make MNIST datasets with spike-encoded data.

    Args:
        encoding_type_label (str): The type of encoding to use (Rate, latency etc). Used to name the cache folder.
        encoding_settings_label (str): The settings for the encoding. Used to name the cache folder.
        enc_function (Callable): The function to use to encode the data.
        batch_size (int): The batch size to use.
        train (bool): Whether to load the training or test set.
        cache_dir (Path): The directory to save the preprocessed data to.
        transforms (torchvision.transforms.Compose): The transforms to apply to the data.
    Returns:
        A Dataset object with spike-encoded data X, y.
    """
    # Sets of the cache dir.
    test_train_prefix = 'train' if train else 'test'
    encoding_settings_label = f"{test_train_prefix}__{encoding_settings_label}"
    cache_dir_mnist_snn = cache_dir / encoding_type_label / encoding_settings_label

    # Set up filename templates.
    prefix = 'train' if train else 'test'
    path_X_template = f'X_{prefix}__batch={{i}}.pt'
    path_y_template = f'y_{prefix}__batch={{i}}.pt'

    # Assume the data has been processed if the cache dir exist.
    if not cache_dir_mnist_snn.exists():
        X, y = get_mnist_processed(train=train, cache_dir=cache_dir, transforms=transforms)

        cache_dir_mnist_snn.mkdir(parents=True)
        range_ = trange if show_progress else range
        for i in range_(0, X.size(0), batch_size):
            # For each batch, encode X, then save X, y to disk.
            idx_start = i
            idx_end = min(i + batch_size, X.size(0))
            X_i = enc_function(X[idx_start:idx_end, :])
            # Swap the axes so (T, N, C) becomes (N, T, C).
            X_i = X_i.permute(1, 0, 2)
            y_i = y[idx_start:idx_end, :]

            T.save(X_i, cache_dir_mnist_snn / path_X_template.format(i=i))
            T.save(y_i, cache_dir_mnist_snn / path_y_template.format(i=i))
    if in_memory:
        return DatasetInMemory(cache_dir_mnist_snn, device=device)
    else:
        return DatasetOnDisk(cache_dir_mnist_snn)


def get_mnist_dataset_spike_encoded__rate(
    batch_size: int,
    num_steps: int,
    train: bool = True,
    cache_dir: Path = Path('./data/'),
    show_progress: bool = False,
    in_memory: bool = False,
    device: T.device = T.device('cpu'),
    transforms: trf.Compose | None = None,
) -> Dataset:
    """Get the MNIST dataset with rate-encoded data."""
    return _get_mnist_dataset_spike_encoded(
        encoding_type_label='mnist_rate_encoded',
        encoding_settings_label=f'batch_size={batch_size}__num_steps={num_steps}',
        enc_function=partial(sg.rate, num_steps=num_steps),  # type: ignore
        batch_size=batch_size,
        train=train,
        cache_dir=cache_dir,
        show_progress=show_progress,
        in_memory=in_memory,
        device=device,
        transforms=transforms,
    )


def get_mnist_dataset_spike_encoded__latency(
    batch_size: int,
    num_steps: int,
    tau: float,
    threshold: float,
    train: bool = True,
    cache_dir: Path = Path('./data/'),
    show_progress: bool = False,
    in_memory: bool = False,
    device: T.device = T.device('cpu'),
    transforms: trf.Compose | None = None,
) -> Dataset:
    """Get the MNIST dataset with latency-encoded data."""
    return _get_mnist_dataset_spike_encoded(
        encoding_type_label='mnist_latency',
        encoding_settings_label=f'batch_size={batch_size}__num_steps={num_steps}__tau={tau:.5f}__threshold={threshold:.5f}',
        enc_function=partial(
            sg.latency,
            num_steps=num_steps,  # type: ignore
            tau=tau,  # type: ignore
            threshold=threshold,
            # clip=True, # Used to clip off spikes at the end of the spike train.
        ),
        batch_size=batch_size,
        train=train,
        cache_dir=cache_dir,
        show_progress=show_progress,
        in_memory=in_memory,
        device=device,
        transforms=transforms,
    )
