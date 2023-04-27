import torch.optim
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, l1_size, l2_size, l3_size):
        self.has_fourth_layer = l3_size > 0
        super().__init__()
        self.l1 = nn.Linear(int(input_size), int(l1_size))
        self.l2 = nn.Linear(int(l1_size), int(l2_size))
        if self.has_fourth_layer:
            self.l3 = nn.Linear(int(l2_size), int(l3_size))
            self.l4 = nn.Linear(int(l3_size), int(output_size))
            nn.init.zeros_(self.l4.weight)
        else:
            self.l3 = nn.Linear(int(l2_size), output_size)
        nn.init.zeros_(self.l1.weight)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l3.weight)

    def forward(self, x):
        x = x.type(torch.float32)
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        if self.has_fourth_layer:
            x = torch.sigmoid(self.l3(x))
            return self.l4(x)
        else:
            result = self.l3(x)
            return result


def get_optimizer(model, learning_rate=0.00025, use_rmsprop=True):
    if use_rmsprop:
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
