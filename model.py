import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabv3(nn.Module):
    def __init__(self):
        super(DeepLabv3, self).__init__()
        self.deep_lab_v3 = models.segmentation.deeplabv3_resnet50(
            weights='COCO_WITH_VOC_LABELS_V1'
        )
        self.deep_lab_v3.classifier = DeepLabHead(2048, 1)

    def forward(self, input):
        output = self.deep_lab_v3(input)['out']
        return output
