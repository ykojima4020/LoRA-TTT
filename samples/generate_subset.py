'''
this is a acript that create a subset of imagenet, comprising around 10% of the total.
'''

import pathlib
import random
import shutil

imanagent_path = pathlib.Path('/path/to/imagenet')
copy_path = pathlib.Path('/path/to/subset')


class_dirs = list(imanagent_path.iterdir())

for class_dir in class_dirs:
    cls = class_dir.parts[-1]
    copy_dir = copy_path / cls 

    files = list(class_dir.iterdir())
    selected_file = random.choice(files)

    print(selected_file)
    shutil.copy(selected_file, copy_dir)
