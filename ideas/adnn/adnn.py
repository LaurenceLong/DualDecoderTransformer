import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AdaptiveNode(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim + feature_dim, feature_dim))
        self.bias = nn.Parameter(torch.randn(feature_dim))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(torch.matmul(x, self.weight) + self.bias)


class ADNN(nn.Module):
    def __init__(self, input_dim, num_nodes, feature_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.nodes = nn.ModuleList([AdaptiveNode(input_dim, feature_dim) for _ in range(num_nodes)])
        self.attention = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.output_layer = nn.Linear(feature_dim, output_dim)

    def forward(self, x, num_iterations=3):
        batch_size = x.shape[0]
        node_states = [torch.zeros(batch_size, self.feature_dim).to(x.device) for _ in self.nodes]
        connections = self.get_connections()

        for _ in range(num_iterations):
            new_states = []
            for i, node in enumerate(self.nodes):
                weighted_states = torch.stack([states * connections[i, j].unsqueeze(0).unsqueeze(1)
                                               for j, states in enumerate(node_states)], dim=0).sum(dim=0)
                node_input = torch.cat([weighted_states, x], dim=1)  # Ensure this matches the expected input
                new_states.append(node(node_input))
            node_states = new_states

        final_state = torch.mean(torch.stack(node_states), dim=0)
        return self.output_layer(final_state)

    def get_connections(self):
        return F.softmax(self.attention, dim=1)

