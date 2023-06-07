import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 1 output, the predicted impact
        )

    def forward(self, input):
        return self.model(input)
