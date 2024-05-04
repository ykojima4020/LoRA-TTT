import torch
import torchvision
import sys
sys.path.append('../')
from evaluator.evaluator import ZeroShotImageNetEvaluator
from evaluator import imagenet_config

from tqdm import tqdm
from misc.utils import AvgMeter

class TestTimeTrainer():

    def __init__(self, data_loader, optimizer, device):
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._device = device
        self._mae_loss_meter = AvgMeter()

    def __call__(self, model):
        model = model.train()
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for idx, (images, target) in enumerate(tqdm_object):
            images = images.to(self._device)
            loss, reconstruction, mask = model(images)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            count = images.size(0)
            self._mae_loss_meter.update(loss.item(), count)
        return self._mae_loss_meter.avg


class TestTimeAdapter():

    def __init__(self, single):
        if single:
            self._ttadapter = SingleSampleAdapter() 
        else:
            self._ttadapter = AllSampleAdapter() 

    def __call__(self, factory, status, config, data_root,
                 num_workers=4, pin_memory=True, device='cuda'):
        # [TODO]: evaluator and trainer should be provided as arguments.
        model, tokenizer, transform = factory.create()
        model = model.to(device)
    
        # [NOTE]: freze parameters not related to TTTPretrainedHFOpenCLIPFactory
        for name, param in model.named_parameters():
            if ('decoder' in name):
                param.requires_grad = False
            if ('text_model.encoder' in name):
                param.requires_grad = False
    
        # [NOTE]: update only image encoder
        if config.optimizer == 'adam':
            eps = 1e-8 
            optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                    eps=eps, lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay)
        elif config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
        else:
            raise TypeError
    
        # [NOTE]: initialization
        dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('valid'))
        evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
        tttrainer = TestTimeTrainer(train_loader, optimizer, device)
    
        # [NOTE]: STEP1: Evaluation of initial model before TTT.
        before_tta = evaluator(model.clip)
        before_tta_top1 = before_tta['eval']['imagenet']['top1']
        before_tta_top5 = before_tta['eval']['imagenet']['top5']
    
        after_tta_top1, after_tta_top5 = self._ttadapter(model, status, transform, tokenizer, optimizer, data_root, config, num_workers, pin_memory, device) 
        return before_tta_top1, before_tta_top5, after_tta_top1, after_tta_top5
    
class AllSampleAdapter():

    def __init__(self):
        pass

    def __call__(self, model, status, transform, tokenizer, optimizer, data_root, config, num_workers, pin_memory, device):
        # [NOTE]: initialization
        dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('valid'))
        evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
        tttrainer = TestTimeTrainer(train_loader, optimizer, device)

        # [NOTE]: after culculation original zero-shot performance, load the finetuned weights.
        model.load_state_dict(status)

        # [NOTE]: STEP2: TTT
        for epoch in range(0, config.epochs):
            tttrainer(model.mae)

        # [NOTE]: STEP3: Evaluation of model after TTT.
        after_tta = evaluator(model.clip, update=False)
        after_tta_top1 = after_tta['eval']['imagenet']['top1']
        after_tta_top5 = after_tta['eval']['imagenet']['top5']
        return after_tta_top1, after_tta_top5


class SingleSampleAdapter():

    def __init__(self):
        pass

    def __call__(self, model, status, transform, tokenizer, optimizer, data_root, config, num_workers, pin_memory, device):
        steps_per_example = config.epochs
        train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('train'))
        train_loader = iter(torch.utils.data.DataLoader(TTTTrainDataset(train_dataset, steps_per_example, config.batch_size), batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory))

        test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('valid')) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        model.load_state_dict(status)
        # [NOTE]: because the text encoder is not updated, text embeddings can be calculated at first.
        text_embeddings = zeroshot_weights(model.clip, tokenizer, imagenet_config.imagenet_classes, imagenet_config.imagenet_templates, device)

        top1, top5, n = 0., 0., 0.

        for test_image, target in tqdm(test_loader):
            test_image = test_image.to(device)
            target = target.to(device)
            model.load_state_dict(status)
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

        after_tta_top1 = (top1 / n) * 100
        after_tta_top5 = (top5 / n) * 100

        return after_tta_top1, after_tta_top5


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
