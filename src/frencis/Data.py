from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder:

    def __init__(self, batch_size=128):
        # Init batch size
        self.batch_size = batch_size

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

    def get_train_loader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def get_test_loader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=True, drop_last=True)
