
import torchvision.transforms as transforms
import timm
import open_clip
import numpy as np

import sys
sys.path.append('../external/robustness/ImageNet-C/imagenet_c')
from imagenet_c import corrupt

def get_original_vit_image_encoder_transforms(mode):
    # the following data_config is given by OriginalViTImageEncoder
    data_config = {'input_size': (3, 224, 224),
                   'interpolation': 'bicubic',
                   'mean': (0.485, 0.456, 0.406),
                   'std': (0.229, 0.224, 0.225),
                   'crop_pct': 0.875,
                   'crop_mode': 'center'}

    if mode == "train":
        return timm.data.create_transform(**data_config, is_training=True)
    else:
        return timm.data.create_transform(**data_config, is_training=False)

def get_open_clip_vitb16_transforms(mode):
    _, train, val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b-b8k')
    if mode == "train":
        return train
    else:
        return val

class Corruption():
    def __init__(self, severity=1, corruption_name=None, corruption_number=-1):
        self._severity = severity
        self._corruption_name = corruption_name
        self._corruption_number = corruption_number

    def __call__(self, x): 
        """
        Args:
            img (PIL Image): Image to be converted.
        """
        x = np.array(x)
        return corrupt(x, severity=self._severity,
                       corruption_name=self._corruption_name,
                       corruption_number=self._corruption_number)

def get_corruption_transform(corruption):
    return transforms.Compose(
            [
            transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None),
            transforms.CenterCrop(size=(224, 224)),
            corruption, 
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])

    # [NOTE]: this is open_clip transform
    # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
