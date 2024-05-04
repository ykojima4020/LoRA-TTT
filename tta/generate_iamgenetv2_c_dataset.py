import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms

from imagenetv2_pytorch import ImageNetV2Dataset
import sys
sys.path.append('../')
from misc.transforms import Corruption

def get_transform(corruption):
    return transforms.Compose(
            [
            transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None),
            transforms.CenterCrop(size=(224, 224)),
            corruption, 
            ])

corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                    'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                    'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

severities = [1, 2, 3, 4, 5]

dataset = ImageNetV2Dataset()

root = Path('/home/ykojima/dataset/imagenetv2-c')

for severity in severities:
    for corruption in corruptions_name:
        # if corruption == 'fog' or corruption == 'frost' or corruption == 'glass_blur':
        #     continue
        store_dir = root / corruption / str(severity)
        store_dir.mkdir(parents=True, exist_ok=True)
        print(store_dir)
        for idx, (x, target) in tqdm(enumerate(dataset)):
            target_dir = store_dir / str(target).zfill(4)
            target_dir.mkdir(exist_ok=True)
 
            transform = get_transform(Corruption(severity=severity, corruption_name=corruption))
            x = transform(x)
            x = Image.fromarray(np.uint8(x)).convert('RGB')

            x.save(target_dir / f'{idx}.jpg', quality=95)

