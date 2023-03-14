import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, ratios, anchor_scales):
        super(RegionProposalNetwork, self).__init__()

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.n_anchors = self.anchor_base.shape[0]

        # Convolutional layer for the shared feature map
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Two separate convolutional layers for predicting the objectness score and bounding box regression
        self.objectness = nn.Conv2d(mid_channels, self.n_anchors*2, kernel_size=1)
        self.regression = nn.Conv2d(mid_channels, self.n_anchors*4, kernel_size=1)

        # Initialize the weights and biases of the layers
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.normal_(self.objectness.weight, std=0.01)
        nn.init.normal_(self.regression.weight, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.objectness.bias, 0)
        nn.init.constant_(self.regression.bias, 0)

    def forward(self, x):
        # Shared convolutional feature map
        shared = F.relu(self.conv(x))

        # Predict the objectness score and bounding box regression for each anchor
        objectness = self.objectness(shared)
        regression = self.regression(shared)

        # Reshape the outputs to have shape (batch_size, n_anchors*2, height, width) and (batch_size, n_anchors*4, height, width), respectively
        objectness = objectness.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 2)
        regression = regression.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)

        # Generate the anchors
        anchors = generate_anchors(self.anchor_base, x.shape[2:], self.n_anchors).to(x.device)

        return objectness, regression, anchors
