import torch as th
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18

background = np.load("background.npy")
bg = th.tensor(background[None, None, :, :], dtype=th.float32).to(device='cuda:0') # 480*640

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 32):
        super(CustomExtractor, self).__init__(observation_space, features_dim)
        
        # Depth
        # self.cnn_depth = nn.Sequential(
        #     nn.Conv2d(2, 4, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(4, 8, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(8, 8, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(1632, 256),
        # )

        self.gripper_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.finger_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.obj_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        combined_feature_dim = 48
        self.fc = nn.Sequential(
            nn.Linear(combined_feature_dim, 32),
        )

    def forward(self, observations):
        # 处理 Depth
        # depth = observations["distance_to_image_plane"][:, 1].squeeze(-1).permute(0, 3, 1, 2)
        # mask = th.abs(depth - bg) > 0.005
        # depth_masked = th.where(mask, depth, th.zeros_like(depth))
        # depth_with_mask = th.cat([depth_masked, mask.float()], dim=1)
        
        # depth_features = self.cnn_depth(depth_with_mask)
        gripper_features = self.gripper_mlp(observations["gripper_pose"][:, :3])  # (num_envs, 7)
        finger_features = self.finger_mlp(observations["finger_state"]) # (num_envs, 2)  
        obj_features = self.obj_mlp(observations["obj_position"]) # (num_envs, 3)
        
        combined_features = th.cat([#depth_features,
                                    obj_features,
                                    gripper_features,
                                    finger_features,
                                    ], dim=1) 

        return self.fc(combined_features)
    

class SecondCustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # RGB
        self.rgb_resnet = resnet18(pretrained=True)
        self.rgb_resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_resnet.fc = nn.Identity()

        # Depth
        self.depth_resnet = resnet18(pretrained=False)
        self.depth_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_resnet.fc = nn.Identity()

        self.gripper_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.finger_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.obj_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        combined_feature_dim = 1072  # RGB + Depth + gripper + finger + obj
        self.fc = nn.Sequential(
            nn.Linear(combined_feature_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb = observations["rgb"][:, 1].permute(0, 3, 1, 2)
        rgb_feat = self.rgb_resnet(rgb)

        depth = observations["distance_to_image_plane"][:, 1].squeeze(-1).permute(0, 3, 1, 2)
        depth_feat = self.depth_resnet(depth)

        gripper_feat = self.gripper_mlp(observations["gripper_pose"][:, :3])
        finger_feat = self.finger_mlp(observations["finger_state"])
        obj_feat = self.obj_mlp(observations["obj_position"])

        combined = th.cat([rgb_feat, depth_feat, obj_feat, gripper_feat, finger_feat], dim=1)
        return self.fc(combined)
