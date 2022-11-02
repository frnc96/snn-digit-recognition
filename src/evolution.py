from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

from datasets import get_mnist_dataset_spike_encoded__rate, get_mnist_dataset_spike_encoded__latency
from src.evaluation import evaluate_accuracy_dataloader, evaluate_loss_dataloader
from src.networks.leaky import LeakyOneLayer as Net
import matplotlib.pyplot as plt
import datetime as dt
import math
import json
import torch as T
import zipfile


class MetricTracker:
    def __init__(
        self, name: str,
        x_label: str,
        y_label: str,
        y_lims: tuple[float, float] | None = None
    ):
        self.name = name
        self.x: list[int] = []
        self.y: list[float] = []
        self.x_label: str = x_label
        self.y_label: str = y_label
        self.y_lims: tuple[float, float] | None = y_lims

    def update(self, generation: int, values: list[float]):
        self.x += [generation for _ in values]
        self.y += values

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "x": self.x,
            "y": self.y
        }


def get_datasets__rate(cache_dir: Path):
    dataset_settings_rate = {
        "batch_size": 4096,
        "num_steps": 16,
        "cache_dir": cache_dir,
        "show_progress": True,
        "in_memory": True,
    }
    print("Setting up data loader for training data...")
    train_dataset = get_mnist_dataset_spike_encoded__rate(train=True, **dataset_settings_rate)
    print("Setting up data loader for testing data...")
    test_dataset = get_mnist_dataset_spike_encoded__rate(train=False, **dataset_settings_rate)

    return train_dataset, test_dataset


def get_datasets__latency(cache_dir: Path):
    dataset_settings_latency = {
        "batch_size": 4096,
        "num_steps": 16,
        "tau": 1,
        "threshold": 0.5,
        "cache_dir": cache_dir,
        "show_progress": True,
        "in_memory": True,
    }

    print("Setting up data loader for training data...")
    train_dataset = get_mnist_dataset_spike_encoded__latency(train=True, **dataset_settings_latency)
    print("Setting up data loader for testing data...")
    test_dataset = get_mnist_dataset_spike_encoded__latency(train=False, **dataset_settings_latency)

    return train_dataset, test_dataset


def main(
    pop_size: int = 16,
    generations: int = 50,
    device: str = 'cuda',
    cache_dir: Path = Path('/mnt/disks/gpu_dev_ssd/data/'),
    save_dir: Path = Path("./outputs/"),
    plot_interval: int = 5,
):
    # train_dataset, test_dataset = get_datasets__rate(cache_dir)
    train_dataset, test_dataset = get_datasets__latency(cache_dir)
    train_dl = DataLoader(train_dataset, batch_size=8000, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_dataset, batch_size=8000, shuffle=True, num_workers=4)

    population: list[Net] = [Net().cuda() for _ in range(pop_size)]

    # Set up a new folder for this run, based on the current date and time
    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = save_dir / now
    code_dir = save_dir / 'code'
    code_dir.mkdir(parents=True, exist_ok=False)
    # Save the contents of ./src/ to the code directory in a zipped folder.
    with zipfile.ZipFile(code_dir / 'src.zip', 'w') as f:
        for file in Path('./src').glob('**/*.py'):
            f.write(file, file.relative_to('./src'))
    metrics_dir = save_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=False)
    models_dir = save_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=False)
    metric_losses = MetricTracker("Loss", "Generation", "Loss")
    metrics_train_accs = MetricTracker("Training Accuracy", "Generation", "Accuracy")
    metrics_test_accs = MetricTracker("Testing Accuracy", "Generation", "Accuracy")
    metrics_mut_rate = MetricTracker("Mutation Rate", "Generation", "Mutation Rate")
    metrics_mut_std = MetricTracker("Mutation Standard Deviation", "Generation", "Mutation Standard Deviation")
    metrics_betas_l1 = MetricTracker("Beta, layer 1", "Generation", "Beta value")
    metrics_thresholds_l1 = MetricTracker("Threshold, layer 1", "Generation", "Threshold value")
    # metrics_betas_l2 = MetricTracker("Beta, layer 2", "Generation", "Beta value")
    # metrics_thresholds_l2 = MetricTracker("Threshold, layer 2", "Generation", "Threshold value")
    metrics: list[MetricTracker] = [
        metric_losses,
        metrics_train_accs,
        metrics_test_accs,
        metrics_mut_rate,
        metrics_mut_std,
        metrics_betas_l1,
        metrics_thresholds_l1,
        # metrics_betas_l2,
        # metrics_thresholds_l2,
    ]

    figure_label = (
        f"Spiking MNIST\n"
        f"Time started: {now}"
        f", generations: {generations}"
        f", #individuals: {pop_size}"
        f"\nNetwork topology: {population[0].describe()}"
    )

    probs = np.array([1 / (i + 1) for i in range(pop_size)]) ** 1
    probs = probs / probs.sum()
    for i_gen in trange(generations+1):
        pop_ranked = evaluate_loss_dataloader(population, train_dl, device)  # type: ignore

        pop_ranked = sorted(pop_ranked, key=lambda x: x.value)
        ranked_population = np.array([result.module for result in pop_ranked])

        population = []
        for i in range(pop_size):
            p1 = np.random.choice(ranked_population, p=probs)

            if np.random.rand() < 0.5:
                p2 = np.random.choice(ranked_population, p=probs)
                child = p1.crossover(p2)
            else:
                child = p1

            if np.random.rand() < 0.5:
                child.mutate()

            population.append(child)
        print()
        print(f"Generation {i_gen}:")
        print(f'\tBest loss: {pop_ranked[0].value}')

        nets = [result.module for result in pop_ranked]
        train_accs = evaluate_accuracy_dataloader([nets[0]], train_dl, device)
        print(f'\tTrain acc: {train_accs[0].value:.5f}')

        test_accs = evaluate_accuracy_dataloader([nets[0]], test_dl, device)
        print(f'\tTest acc: {test_accs[0].value:.5f}')

        metrics_train_accs.update(i_gen, [result.value for result in train_accs])
        metrics_test_accs.update(i_gen, [result.value for result in test_accs])
        metric_losses.update(i_gen, [result.value for result in pop_ranked])
        metrics_mut_rate.update(i_gen, [float(result.module.mutation_rate.item()) for result in pop_ranked])  # type: ignore
        metrics_mut_std.update(i_gen, [float(result.module.mutation_std.item()) for result in pop_ranked])  # type: ignore
        metrics_betas_l1.update(i_gen, [float(result.module.lif1.beta.item()) for result in pop_ranked])  # type: ignore
        metrics_thresholds_l1.update(i_gen, [float(result.module.lif1.threshold.item()) for result in pop_ranked])  # type: ignore
        # metrics_betas_l2.update(i_gen, [float(result.module.lif2.beta.item()) for result in pop_ranked])  # type: ignore
        # metrics_thresholds_l2.update(i_gen, [float(result.module.lif2.threshold.item()) for result in pop_ranked])  # type: ignore

        if i_gen % plot_interval == 0:
            grid_size = math.ceil(math.sqrt(len(metrics)))
            fig, axes = plt.subplots(ncols=grid_size, nrows=grid_size, figsize=(8*grid_size, 8*grid_size))
            fig.suptitle(figure_label, x=0.01, y=0.99, horizontalalignment='left', verticalalignment='top')
            ax: plt.Axes
            for metric, ax in zip(metrics, axes.flat):  # type: ignore
                ax.scatter(metric.x, metric.y, s=5)
                ax.set_title(metric.name)
                ax.set_xlabel(metric.x_label)
                ax.set_ylabel(metric.y_label)
                ax.set_xlim(0, i_gen + 1)
                if metric.y_lims is not None:
                    ax.set_ylim(*metric.y_lims)

            plot_path = plots_dir / f'plot_gen={i_gen:04d}.png'
            fig.savefig(str(plot_path))

            for metric in metrics:
                # Pad the name with 4 0s.
                metrics_path = metrics_dir / f'metrics__gen={i_gen:04d}__{metric.name}.json'
                with open(metrics_path, 'w') as f:
                    json.dump(metric.to_dict(), f, indent=4)

            model_path = models_dir / f'model__{i_gen:04d}__acc={100*test_accs[0].value:.8f}%__loss={pop_ranked[0].value:.8f}.pt'
            T.save(pop_ranked[0].module.state_dict(), model_path)


if __name__ == '__main__':
    raise SystemExit(main())
