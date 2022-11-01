from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

from datasets import get_mnist_dataset_spike_encoded__rate
from src.evaluation import evaluate_accuracy_dataloader, evaluate_loss_dataloader
from src.networks import Net


def main(
    pop_size: int = 16,
    generations: int = 1000,
    device: str = 'cuda',
):
    print("Setting up data loader for training data...")
    train_dataset = get_mnist_dataset_spike_encoded__rate(
        4096,
        32,
        cache_dir=Path('/mnt/disks/gpu_dev_ssd/data/'),
        show_progress=True,
        train=True,
        in_memory=True
    )
    # train_dataset = get_mnist_dataset_spike_encoded__latency(
    #     batch_size=4096,
    #     num_steps=32,
    #     tau=1,
    #     threshold=0.5,
    #     train=True,
    #     cache_dir=Path('/mnt/disks/gpu_dev_ssd/data/'),
    #     show_progress=True,
    #     in_memory=True,
    # )
    train_dl = DataLoader(train_dataset, batch_size=8000, shuffle=True, num_workers=0)

    print("Setting up data loader for testing data...")
    test_dataset = get_mnist_dataset_spike_encoded__rate(
        4096,
        32,
        cache_dir=Path('/mnt/disks/gpu_dev_ssd/data/'),
        show_progress=True,
        train=False,
        in_memory=True
    )
    # test_dataset = get_mnist_dataset_spike_encoded__latency(
    #     batch_size=4096,
    #     num_steps=32,
    #     tau=0.5,
    #     threshold=0.5,
    #     train=False,
    #     cache_dir=Path('/mnt/disks/gpu_dev_ssd/data/'),
    #     show_progress=True,
    #     in_memory=True,
    # )
    test_dl = DataLoader(test_dataset, batch_size=8000, shuffle=True, num_workers=0)

    population: list[Net] = [Net().cuda() for _ in range(pop_size)]

    metric_losses = []

    probs = np.array([1 / (i + 1) for i in range(pop_size)]) ** 1.2
    probs = probs / probs.sum()
    for i_gen in trange(generations):
        pop_ranked = evaluate_loss_dataloader(population, train_dl, device)  # type: ignore

        pop_ranked = sorted(pop_ranked, key=lambda x: x[0])
        metric_losses += []
        ranked_population = np.array([net for _, net in pop_ranked])

        population = []
        for i in range(pop_size):
            p1 = np.random.choice(ranked_population, p=probs)

            if np.random.rand() < 0.2:
                child = p1
            else:
                p2 = np.random.choice(ranked_population, p=probs)
                child = p1.crossover(p2)

            if np.random.rand() < 0.5:
                child.mutate()

            population.append(child)
        print()
        print(f'Generation {i_gen} best loss: {pop_ranked[0][0]}')

        acc = evaluate_accuracy_dataloader(pop_ranked[0][1], train_dl, device)
        print(f'Generation {i_gen} train acc: {acc:.5f}')

        acc = evaluate_accuracy_dataloader(pop_ranked[0][1], test_dl, device)
        print(f'Generation {i_gen} test acc: {acc:.5f}')


if __name__ == '__main__':
    raise SystemExit(main())
