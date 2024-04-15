import argparse
import json
import sys
import pathlib
import pandas as pd

def get_coco_captions_df(caption_json):

    with open(caption_json, "r") as f:
        annotations = json.load(f)

    image_dict = {}
    for image in annotations['images']:
        image_dict[image['id']] = image['file_name']

    image_names = []
    captions = []
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        image_names.append(image_dict[image_id])
        captions.append(caption)

    df = pd.DataFrame({'image': image_names,
                       'caption': captions})
    return df

def get_coco_captions_test_df(caption_json, length=100):
    df = get_coco_captions_df(caption_json)
    return df[:length]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('caption_json')
    # parser.add_argument('--output', default='./train_df')
    args = parser.parse_args()

    df = get_coco_captions_df(args.caption_json)

