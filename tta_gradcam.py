import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from factory import PretrainedHFOpenCLIPFactory
from tta import MEMLoss, zeroshot_weights, build_tta_optimizer, accuracy
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from misc.config import load_config
from omegaconf import OmegaConf
from misc.tpt_transforms import AugMixAugmenter

def show_cam_on_image(img, mask, neg_saliency=False):

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def main():

    # configurations
    config = './config/mae_clip_run.yaml'
    device = 'cuda'
    config = load_config(config)
    OmegaConf.set_struct(config, True)

    factory = PretrainedHFOpenCLIPFactory(config.model, mae=config.reconst)
    model, tokenizer, transform = factory.create()
    model = model.to(device)
    # without fine-tuning
    init_status = model.mae.state_dict()

    # [NOTE]: Data augmentation
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=BICUBIC),
                    transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
    batch_size = 64
    tta_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1,
                                   augmix=False)


    # optimizer
    optimizer = build_tta_optimizer(model, config.tta['peft'])

    # Loss
    loss = MEMLoss()
    tta_runner = SingleImageLoRATTARunner(config.tta['peft'], loss) 

    # inferencer
    classifier = TruthClassifier()

    # dataset
    image_path = '/data2/yuto/dataset/imagenetv2-c/snow/5/0000/4803.jpg'
    target = torch.tensor([0])
    image = Image.open(image_path)
    images = tta_transform(image)

    data_root = '/data2/yuto/dataset/imagenetv2-c/snow/5'
    save_dir = Path(f'./gradcam/snow/{loss.__class__.__name__}/')
    save_dir.mkdir(parents=True, exist_ok=True)
    before_save_dir = save_dir / 'before'
    after_save_dir = save_dir / 'after'
    before_save_dir.mkdir(parents=True, exist_ok=True)
    after_save_dir.mkdir(parents=True, exist_ok=True)

    tta_data = ImageFolderWithPaths(root=data_root, transform=tta_transform)

    classes = imagenet_classes
    prompts = ensemble_prompts

    # text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)
    # text_embeddings = text_embeddings.cpu()
    # torch.save(text_embeddings, './text_embeddings.pth')

    text_embeddings = torch.load('./text_embeddings.pth')
    text_embeddings = text_embeddings.to(device)


    for i, (images_tmp, target_tmp, path) in tqdm(enumerate(tta_data)):
        p = Path(path)
        file_name = p.name
        cls = p.parent.parts[-1]
        before_cls_save_dir = before_save_dir / cls
        after_cls_save_dir = after_save_dir / cls
        before_cls_save_dir.mkdir(parents=True, exist_ok=True)
        after_cls_save_dir.mkdir(parents=True, exist_ok=True)

 
        for k in range(len(images)):
            images[k] = images[k].to(device)
        target = torch.tensor([target]).to(device)
        image = images[0].unsqueeze(0)
        images = torch.stack(images)

        model.mae.load_state_dict(init_status)

        # inference before TTA
        before_acc = classifier(model, image, text_embeddings, target) 
        before_image = clip_grad_cam(model, image, target[0], text_embeddings)
        before_image.save(before_save_dir / cls / file_name)

        # single image TTA
        tta_runner(model, init_status, images, text_embeddings, optimizer)
 
        # inference after TTA
        after_acc = classifier(model, image, text_embeddings, target) 
        after_image = clip_grad_cam(model, image, target[0], text_embeddings)
        after_image.save(after_save_dir / cls / file_name)

        print(file_name, before_acc, after_acc)


class SingleImageLoRATTARunner():

    def __init__(self, config, loss):
        print(f'{self} created.')
        print(config)
        self._config = config

        if isinstance(loss, MEMLoss):
            self._loss = loss
        elif isinstance(loss, MAELoss):
            self._loss = loss
        elif isinstance(loss, MAEMEMLoss):
            self._loss = loss
        else:
            raise TypeError

    def __call__(self, model, status, images, text_embeddings, optimizer):
        for name, param in model.image_encoder.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        # [TODO]: should load only LoRA and Decoder, not update text_encoder
        model.mae.load_state_dict(status)
        model.train()
        for j in range(1): # epoch is once
            l = self._loss(model, images, text_embeddings)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        model.zero_grad()

class TruthClassifier():

    def __init__(self):
        pass

    def __call__(self, model, image, text_embeddings, target):
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features = model.clip.image_encode(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                output = image_features @ text_embeddings
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        res = True if acc1 == 100.0 else False
        return res

def clip_grad_cam(model, image, target, text_embeddings):

    model.train()
    # [NOTE]: the model shoud be trainable
    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = True
    
    image_feature = model.clip.image_encode(image)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

    similarity = image_feature @ text_embeddings
    objective = similarity[0][target]

    # [NOTE]: calculate gradiation of the intermediate tensor
    attention_weight = model.image_encoder.get_attn()[11]
    attention_weight.retain_grad()
    
    model.zero_grad()
    objective.backward()

    # [NOTE]: GradCAM calculation
    last_attn = attention_weight.detach()
    last_attn = last_attn.reshape(-1, last_attn.shape[-1], last_attn.shape[-1])

    last_grad = attention_weight.grad.detach()
    last_grad = last_grad.reshape(-1, last_grad.shape[-1], last_grad.shape[-1])

    cam = last_grad * last_attn
    cam = cam.clamp(min=0).mean(dim=0)
    image_relevance = cam[0, 1:]

     # [NOTE]: create image
    image_relevance = image_relevance.reshape(1, 1, 14, 14)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance, neg_saliency=False)
    vis = vis[..., ::-1] # BGR RGB convert

    # [NOTE]: the model shoud be froze 
    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = False

    model.zero_grad()
    return Image.fromarray(np.uint8(255 * vis))


from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        # 標準の ImageFolder データセットの機能を使用してデータとラベルを取得
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # データパスを取得
        path = self.imgs[index][0]
        # データ、ラベル、パスを返す
        return original_tuple + (path,)

if __name__ == "__main__":
    main()

