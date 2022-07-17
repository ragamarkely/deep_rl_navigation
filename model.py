import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class QNetwork(nn.Module):
    """
    Policy Model
    """

    def __init__(
        self, 
        state_size: int, 
        action_size: int, 
        seed: int, 
        fc1_units: int=64, 
        fc2_units: int=64,
    ) -> None:
        """
        Initialize  hyperparameters and build model.

        Params
        ======
            state_size: number of states
            action_size: number of actions
            seed: random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state: np.ndarray) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)