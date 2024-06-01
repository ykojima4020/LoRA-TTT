
origin_classes = '../evaluator/classnames.txt'
target_ids = '/home/ykojima/dataset/imagenet-a/ids.txt'

with open(origin_classes, 'r') as f:
    lines = f.readlines()

id_classes_dict = {}
all_classes = []
for line in lines:
    id_and_class = line.split()
    id_classes_dict[id_and_class[0]] = ' '.join(id_and_class[1:])
    all_classes.append(' '.join(id_and_class[1:]))

print(id_classes_dict)
print(all_classes)


with open(target_ids, 'r') as f:
    lines = f.readlines()

target_classes = []
for line in lines:
    id = line.split()[0]
    cls = id_classes_dict[id]
    target_classes.append(cls)

print(target_classes)