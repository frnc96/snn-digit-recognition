import torch
import torch.nn as nn
import numpy as np

# todo - extract this in a constant
batch_size = 128

num_epochs = 1
loss_hist = []
test_loss_hist = []
dtype = torch.float
num_steps = 25


class BackpropTT:

    def __init__(self, net):
        self.net = net
        self.device = next(net.parameters()).device
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    def print_batch_accuracy(self, data, targets, train=False):
        output, _ = self.net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
            print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")

    def train_printer(self, epoch, iter_counter, data, targets, test_data, test_targets, counter):
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        self.print_batch_accuracy(data, targets, train=True)
        self.print_batch_accuracy(test_data, test_targets, train=False)
        print("\n")

    def training_loop(self, train_loader, test_loader):
        counter = 0

        # Outer training loop
        for epoch in range(num_epochs):
            iter_counter = 0
            train_batch = iter(train_loader)

            # Minibatch training loop
            for data, targets in train_batch:
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward pass
                self.net.train()
                spk_rec, mem_rec = self.net(data.view(batch_size, -1))

                # initialize the loss & sum over time
                loss_val = torch.zeros(1, dtype=dtype, device=self.device)
                for step in range(num_steps):
                    loss_val += self.loss(mem_rec[step], targets)

                # Gradient calculation + weight update
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                # Test set
                with torch.no_grad():
                    self.net.eval()
                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.to(self.device)
                    test_targets = test_targets.to(self.device)

                    # Test set forward pass
                    test_spk, test_mem = self.net(test_data.view(batch_size, -1))

                    # Test set loss
                    test_loss = torch.zeros(1, dtype=dtype, device=self.device)
                    for step in range(num_steps):
                        test_loss += self.loss(test_mem[step], test_targets)
                    test_loss_hist.append(test_loss.item())

                    # Print train/test loss/accuracy
                    if counter % 50 == 0:
                        self.train_printer(epoch, iter_counter, data, targets, test_data, test_targets, counter)
                    counter += 1
                    iter_counter += 1
