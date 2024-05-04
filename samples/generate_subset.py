'''
this is a acript that create a subset of imagenet, comprising around 10% of the total.
'''

import pathlib
import random
import shutil

imanagent_path = pathlib.Path('/home/ykojima/dataset/imagenetv2-c/original/0')
copy_path = pathlib.Path('/home/ykojima/dataset/imagenetv2-c/original_10/0')


class_dirs = list(imanagent_path.iterdir())

for class_dir in class_dirs:
    cls = class_dir.parts[-1]
    copy_dir = copy_path / cls 

    files = list(class_dir.iterdir())
    selected_file = random.choice(files)

    print(copy_dir)
    print(selected_file)
    shutil.copy(selected_file, copy_dir)
