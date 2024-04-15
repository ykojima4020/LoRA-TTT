'''
This script aims to do test time training proposed by Test-Time Training with Masked Autoencoders.
This implementation tries to be the same as that.
'''

import argparse
import torch
import torchvision
import wandb
from tqdm import tqdm

import sys
sys.path.append('../')
from factory import RILSMAECLIPFactory, PretrainedOpenCLIPDecoderEncoderFineTuneFactory, PretrainedHFOpenCLIPFactory
from evaluator import imagenet_config
from misc.config import load_config
from omegaconf import OmegaConf

columns = ['layer', 'lr', 'weight_decay', 'batch_size', 'epochs', 'optimizer', 'severity', 'corruption',
           'top1', 'top5']

def get_args_parser():
    parser = argparse.ArgumentParser('Test Time Training', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--type', default='open', choices=['normal', 'open', 'hf_open'], help='a kind of archtectures')
    parser.add_argument('--reconst', default='feature', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    return parser

def main():
    args = get_args_parser()
    args = args.parse_args()

    # fixed parameters
    eps = 1e-8 
    device = 'cuda'
    num_workers = 4
    pin_memory = True

    config = load_config(args.cfg)
    OmegaConf.set_struct(config, True)    

    if args.wandb:
        run = wandb.init(project='mae_clip_ttt',
                         entity="ykojima",
                         config=OmegaConf.to_container(config, resolve=True))
        config = OmegaConf.create(dict(wandb.config))
    print(config)

    # [NOTE]: this table is for logging
    table = wandb.Table(columns=columns)

    if args.type == 'normal':
        factory = RILSMAECLIPFactory(config.model)
    elif args.type == 'open':
        factory = PretrainedOpenCLIPDecoderEncoderFineTuneFactory(config.model, mae=args.reconst)
    elif args.type == 'hf_open':
        factory = PretrainedHFOpenCLIPFactory(config.model, mae=args.reconst)
    else:
        raise TypeError

    model, tokenizer, transform = factory.create()
    model = model.to(device)
    status = torch.load(args.checkpoint, map_location="cuda")

    # [NOTE]: freze parameters not related to TTT
    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = False
        if ('text_model.encoder' in name):
            param.requires_grad = False
        print(name, param.requires_grad)

    # [NOTE]: update only image encoder
    if config.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                eps=eps, lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
    else:
        raise TypeError

    steps_per_example = config.epochs

    severity = 5
    corruption = 'contrast'
    train_dataset = torchvision.datasets.ImageFolder(root=f'~/dataset/imagenetv2-c/{corruption}/{severity}', transform=transform('train'))
    train_loader = iter(torch.utils.data.DataLoader(TTTTrainDataset(train_dataset, steps_per_example, config.batch_size), batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory))

    test_dataset = torchvision.datasets.ImageFolder(root=f'~/dataset/imagenetv2-c/{corruption}/{severity}', transform=transform('valid')) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model.load_state_dict(status['model'])
    text_embeddings = zeroshot_weights(model.clip, tokenizer, imagenet_config.imagenet_classes, imagenet_config.imagenet_templates, device)

    top1, top5, n = 0., 0., 0.

    for test_image, target in tqdm(test_loader):
        test_image = test_image.to(device)
        target = target.to(device)
        model.load_state_dict(status['model'])
        model.train()
        # [NOTE]: optimizer should be initialized? => don't have to initialize because lr is constant.

        for step_per_example in range(steps_per_example): 
            train_data = next(train_loader)
            train_image, _ = train_data
            train_image = train_image.to(device)
            loss, reconstruction, mask = model.mae(train_image) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 

        model.eval()
        with torch.no_grad():
            # predict
            image_features = model.clip.image_encode(test_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_embeddings

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += test_image.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    table.add_data(config.layer, config.lr, config.weight_decay, config.batch_size, config.epochs, config.optimizer, severity, corruption, top1, top5)
    print(table.get_dataframe())
    table.get_dataframe().to_csv(args.output, index=False)
    print(top1, top5)

    if args.wandb:
        wandb.log({'result': table})
 
class TTTTrainDataset():
    '''
    this dataset classs is based on https://github.com/yossigandelsman/test_time_training_mae/blob/main/data/tt_image_folder.py#L7
    '''

    def __init__(self, dataset, steps_per_example, batch_size):
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError
        self.dataset = dataset
        self.batch_size = batch_size 
        self.steps_per_example = steps_per_example

    def __len__(self):
        return self.batch_size * self.steps_per_example * len(self.dataset) 

    def __getitem__(self, index):
        real_index = (index // (self.steps_per_example * self.batch_size))
        image, target = self.dataset[real_index]
        return image, target


def zeroshot_weights(model, tokenizer, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # 80 patterns per class
            texts = [template.format(classname) for template in templates] #format with class
            max_length = 15
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
            batch = {key: values.to(device) for key, values in tokens.items()}
            class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
            class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == "__main__":
    main()

