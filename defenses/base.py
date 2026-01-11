





import torch.nn as nn


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'EmptyDefense (Identity)'
