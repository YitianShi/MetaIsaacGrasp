# http://skrl.readthedocs.io/en/latest/api/models/gaussian.html#models-gaussian

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class PolicyMLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class ValueMLP(Model):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def act(self, inputs, role): # act!!!
        return self.net(inputs["states"]), {}, {}
