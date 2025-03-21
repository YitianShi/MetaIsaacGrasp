# 理想情况是，这里可以直接对obs_buf进行处理
# 输入：rgbd+pcd
# 输出：(env, features_dim)
import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(CustomExtractor, self).__init__(observation_space, features_dim)

        # # RGB
        # self.cnn_rgb = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        
        # Depth
        self.cnn_depth = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.gripper_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )


        # PCD 这里整个都要改！不能用普通卷积！
        # self.cnn_pcd = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        combined_feature_dim = 6592 # 13056 + 6528 + 6528 尬住了，先写死吧，之后改成自动计算的
        self.fc = nn.Sequential(
            nn.Linear(combined_feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # os.makedirs("images", exist_ok=True)
        # for key, value in observations.items():
        #     if key == "distance_to_image_plane":
        #         observations[key] = torch.nan_to_num(value, nan=0.0)

        #         images = observations[key][:, 1].squeeze(-1).cpu().numpy()

        #         batch_size = images.shape[0]
        #         fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
        #         if batch_size == 1: axes = [axes]

        #         for i in range(batch_size):
        #             axes[i].imshow(images[i])
        #             axes[i].set_title(f"Depth_Image(Env {i})")
        #             axes[i].axis("off")

        #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #         image_path = os.path.join("images", f"depth_{timestamp}.png")
        #         plt.savefig(image_path)
        #         plt.close(fig)
        
        # 处理 RGB
        # rgb = observations["rgb"][:, 0].permute(0, 3, 1, 2) # (num_envs, 3, 480, 640)
        # rgb = torch.nan_to_num(rgb, nan=0.0) # rgb有Nan值，现在的处理是变成0,后续应该检查为什么出现这种情况
        # rgb_features = self.cnn_rgb(rgb)
        # 处理 Depth
        depth_features = self.cnn_depth(observations["distance_to_image_plane"].squeeze(1).squeeze(-1).permute(0, 3, 1, 2))  # (num_envs, 1, 480, 640)
        # 处理 PCD
        # pcd_features = self.cnn_pcd(observations["pcd"][:, 0].permute(0, 3, 1, 2))  # (num_envs, 3, 480, 640)
        # 拼接特征
        # features = torch.cat([rgb_features, depth_features, pcd_features], dim=1)
        gripper_features = self.gripper_mlp(observations["gripper_pose"])  # (num_envs, 7)
        combined_features = torch.cat([depth_features, gripper_features], dim=1) 

        return self.fc(combined_features)
