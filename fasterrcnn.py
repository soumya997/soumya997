
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()

        # Load the pre-trained backbone network
        self.backbone = torchvision.models.vgg16(pretrained=True)

        # Remove the fully connected layers
        self.backbone = nn.Sequential(*list(self.backbone.features.children())[:-1])

        # RPN
        self.rpn = RegionProposalNetwork(in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32])

        # ROI Pooling
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)

        # Classifier and Bounding Box Regression
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes+1),
        )

        self.bbox_reg = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4*(num_classes+1)),
        )

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)

        # RPN
        rpn_scores, rpn_locs, anchors = self.rpn(features)

        # Proposal generation
        proposals = generate_proposals(rpn_scores, rpn_locs, anchors)

        # ROI Pooling
        roi_features = self.roi_pool(features, proposals)

        # Classifier and Bounding Box Regression
        flatten = torch.flatten(roi_features, start_dim=1)
        scores = self.classifier(flatten)
        bbox_deltas = self.bbox_reg(flatten)

        return scores, bbox_deltas, proposals
