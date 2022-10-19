import torch
import torch.nn as nn
import numpy as np
import src.domain.constants.parameters as params

dtype = torch.float


def print_batch_accuracy(
        net,
        data,
        batch_size,
        targets,
        train=False
):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")


def train_printer(
        net,
        epoch,
        iter_counter,
        data,
        targets,
        test_data,
        test_targets,
        loss_hist,
        test_loss_hist,
        counter
):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(net, data, params.BATCH_SIZE, targets, train=True)
    print_batch_accuracy(net, test_data, params.BATCH_SIZE, test_targets, train=False)
    print("\n")


class BackpropTT:
    loss_hist = []
    test_loss_hist = []

    def __init__(self, net):
        self.net = net
        self.device = next(net.parameters()).device
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    def train(self, train_loader, test_loader):
        counter = 0

        # Outer training loop
        for epoch in range(params.NUM_OF_EPOCHS):
            iter_counter = 0
            train_batch = iter(train_loader)

            # Minibatch training loop
            for data, targets in train_batch:
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.net.train()
                spk_rec, mem_rec = self.net(data.view(params.BATCH_SIZE, -1))

                # Initialize the loss & sum over time
                loss_val = torch.zeros(1, dtype=dtype, device=self.device)
                for step in range(params.NUM_OF_STEPS):
                    loss_val += self.loss(mem_rec[step], targets)

                # Gradient calculation + weight update
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                # Store loss history for future plotting
                self.loss_hist.append(loss_val.item())

                # Test set
                with torch.no_grad():
                    self.net.eval()
                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.to(self.device)
                    test_targets = test_targets.to(self.device)

                    # Test set forward pass
                    test_spk, test_mem = self.net(test_data.view(params.BATCH_SIZE, -1))

                    # Test set loss
                    test_loss = torch.zeros(1, dtype=dtype, device=self.device)
                    for step in range(params.NUM_OF_STEPS):
                        test_loss += self.loss(test_mem[step], test_targets)
                    self.test_loss_hist.append(test_loss.item())

                    # Print train/test loss/accuracy
                    if counter % 50 == 0:
                        train_printer(
                            net=self.net,
                            epoch=epoch,
                            iter_counter=iter_counter,
                            data=data,
                            targets=targets,
                            test_data=test_data,
                            test_targets=test_targets,
                            loss_hist=self.loss_hist,
                            test_loss_hist=self.test_loss_hist,
                            counter=counter
                        )
                    counter += 1
                    iter_counter += 1

    def test(self, test_loader):
        total = 0
        correct = 0

        # Iterate through all mini-batches and measure accuracy over the full data set
        with torch.no_grad():
            self.net.eval()
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward pass
                test_spk, _ = self.net(data.view(data.size(0), -1))

                # calculate total accuracy
                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            accuracy = 100 * (correct / total)

            print(f"Total correctly classified test set images: {correct}/{total}")
            print(f"Test Set Accuracy: {accuracy:.2f}%")

            return round(accuracy)
