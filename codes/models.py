
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
# from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim = [1024, 512], hidden_layers = 2):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_dim[0])
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[-1])
        self.layer3 = nn.Linear(hidden_dim[-1], output_size)
        self.norm1 = nn.BatchNorm1d(hidden_dim[0])
        self.norm2 = nn.BatchNorm1d(hidden_dim[-1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout1d(0.2)
        self.sig = nn.Sigmoid()
    def forward(self,xs):
        xs = self.relu(self.norm1(self.layer1(xs)))
        xs = self.relu(self.norm2(self.layer2(xs)))
        xs = self.relu(self.layer3(xs))
        return xs
    
class Critic(nn.Module):
    def __init__(self, input_size, hidden_dim = 512, hidden_layers = 2):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.classifier = nn.Linear(hidden_dim//2, 1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim//2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout1d(0.2)
        self.sig = nn.Sigmoid()
    def forward(self, xs):
        xs = self.relu(self.norm1(self.layer1(xs)))
        xs = self.relu(self.norm2(self.layer2(xs)))
        xs = self.classifier(xs)
        return xs