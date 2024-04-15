import os
import cv2
import torch
from torchvision import transforms
import timm

from PIL import Image

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self._image_path = image_path 
        self.image_filenames = image_filenames
        self.captions = list(captions)
        max_length = 200 
        # [NOTE]: different tokenizer interface 
        self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        # [NOTE]: different tokenizer interface 
        item = {'input_ids': self.encoded_captions['input_ids'][idx],
                'attention_mask': self.encoded_captions['attention_mask'][idx]}

        image = Image.open(f"{self._image_path}/{self.image_filenames[idx]}") 
        if image.mode == 'L':
            image = image.convert('RGB') # This is because there's images in a gray scale.
        image = self.transforms(image)
        item['image'] = image
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

