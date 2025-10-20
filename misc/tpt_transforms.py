# This file is based on https://github.com/azshue/TPT/blob/main/data/datautils.py#L61-L102. 
# -------------------------------------------------------------------------
# Written by Yuto Kojima
# -------------------------------------------------------------------------

from PIL import Image
import torchvision.transforms as transforms

# AugMix Transforms
def get_preaugment(image_size=224):
    return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, image_size=224, severity=1):
    preaugment = get_preaugment(image_size)
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix 
    return mix 


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    image_size=224, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.image_size = image_size
        self.severity = severity
    
    def __call__(self, x): 
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.image_size, self.severity) for _ in range(self.n_views)]
        return [image] + views
