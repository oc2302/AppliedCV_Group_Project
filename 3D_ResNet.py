import torch

model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

device = "cpu"
model = model.eval()
model = model.to(device)

