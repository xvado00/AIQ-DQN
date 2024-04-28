import torch.optim
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, linear_layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        normal_function = nn.init.xavier_normal_

        last_layer_size = 0
        for i in range(len(linear_layer_sizes)):
            if linear_layer_sizes[i] <= 0:
                break
            prev_size = linear_layer_sizes[i - 1] if (i - 1) >= 0 else input_size
            self.layers.append(
                nn.Linear(int(prev_size), int(linear_layer_sizes[i]))
            )
            last_layer_size = int(linear_layer_sizes[i])
        self.layers.append(
            nn.Linear(last_layer_size, int(output_size))
        )


        for layer in self.layers:
            normal_function(layer.weight)


    def forward(self, x):
        x = x.type(torch.float32)

        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))

        return self.layers[-1](x)


def get_optimizer(model, learning_rate=0.00025, use_rmsprop=True):
    if use_rmsprop and False:
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=0.95,
            eps=0.01
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        amsgrad=True,
    )


def get_criterion(reduction='none'):
    return nn.SmoothL1Loss(reduction=reduction)
