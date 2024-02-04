import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    config.d_model = 512
    config.d_ff = 2048
    config.dropout = 0.1
    ff = FeedForward(config)
    x = torch.randn(2, 5, config.d_model)
    print(ff(x).size()) # torch.Size([2, 5, 512])
    print(ff)