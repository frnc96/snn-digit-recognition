from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import src.domain.constants.parameters as params


class Encoder:

    def __init__(self):
        # Define a transform
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        # Download training data
        self.mnist_train = datasets.MNIST(root="../data", train=True, download=True, transform=self.transform)

        # Download test data
        self.mnist_test = datasets.MNIST(root="../data", train=False, download=True, transform=self.transform)

    def get_loaders(self):
        train_loader = DataLoader(self.mnist_train, batch_size=params.BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(self.mnist_test, batch_size=params.BATCH_SIZE, shuffle=True, drop_last=True)

        return train_loader, test_loader
