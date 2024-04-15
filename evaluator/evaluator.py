import torch
from tqdm import tqdm

from evaluator import imagenet_config

class Evaluator():

    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

class ZeroShotImageNetEvaluator(Evaluator):

    def __init__(self, tokenizer, device, dataset, batch_size=32, num_workers=2):
        self._tokenizer = tokenizer
        self._imagenet_classes = imagenet_config.imagenet_classes
        self._imagenet_templates = imagenet_config.imagenet_templates

        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError('{} is not supported.'.format(type(dataset)))

        self._loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self._device = device
        self._zeroshot_weights = None

    def _zeroshot_classifier(self, model, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                # 80 patterns per class
                texts = [template.format(classname) for template in templates] #format with class
                max_length = 15
                tokens = self._tokenizer(texts, padding=True, truncation=True, max_length=max_length)
                batch = {key: values.to(self._device) for key, values in tokens.items()}
                class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
                class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def _accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
 
    def __call__(self, model, update=True):
        if update or (self._zeroshot_weights is None):
            self._zeroshot_weights = self._zeroshot_classifier(model, self._imagenet_classes, self._imagenet_templates)
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(self._loader)):
                images = images.to(self._device)
                target = target.to(self._device)
                
                # predict
                image_features = model.image_encode(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ self._zeroshot_weights
        
                # measure accuracy
                acc1, acc5 = self._accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
        
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 
        return {'eval': {'imagenet': {'top1': top1,
                                      'top5': top5}}} 