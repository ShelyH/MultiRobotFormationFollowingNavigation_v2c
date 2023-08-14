import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model,model1,model2, memory, batch_size):
        self.model = model
        self.model1 = model1
        self.model2 = model2
        self.criterion = nn.MSELoss()
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.optimizer1 = None
        self.optimizer2 = None

    def set_learning_rate(self, learning_rate):
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer1 = optim.SGD(self.model1.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer2 = optim.SGD(self.model2.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values, inputs1, values1, inputs2, values2 = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)
            inputs1 = Variable(inputs1)
            values1 = Variable(values1)
            inputs2 = Variable(inputs2)
            values2 = Variable(values2)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()

            self.optimizer1.zero_grad()
            outputs1 = self.model1(inputs1)
            loss1 = self.criterion1(outputs1, values1)
            loss1.backward()
            self.optimizer1.step()

            self.optimizer2.zero_grad()
            outputs2 = self.model2(inputs2)
            loss2 = self.criterion2(outputs2, values2)
            loss2.backward()
            self.optimizer2.step()
            losses += loss.data.item()

        average_loss = losses / num_batches

        return average_loss
