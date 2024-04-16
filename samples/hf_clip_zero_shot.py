import sys
sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory

from transformers import CLIPModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPProcessor

from PIL import Image
from misc.config import load_config
from omegaconf import OmegaConf
import numpy as np

from evaluator.evaluator import ZeroShotImageNetEvaluator
from imagenetv2_pytorch import ImageNetV2Dataset

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
image_processor = processor.image_processor
tokenizer = processor.tokenizer

target_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

image = Image.open('/home/ykojima/dataset/imagenet-c/speckle_noise/1/n02106550/ILSVRC2012_val_00022664.JPEG')
inputs = processor(text=["a phot of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt")

outputs = target_model(**inputs)

print('----target----')
print(outputs.vision_model_output.pooler_output.shape)
print(outputs.vision_model_output.pooler_output[0][0])

logits_per_image = outputs.logits_per_image
target_probs = logits_per_image.softmax(dim=1)
print("target_probs: ", target_probs)

clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
outputs = clip_vision_model(inputs['pixel_values'])
print(outputs.pooler_output.shape)
print(outputs.pooler_output[0][0])

print('----original----')
config = '../config/ttt_mae.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)
factory = PretrainedHFOpenCLIPFactory(cfg.model)

model, tokenizer, transform = factory.create()
model.eval()

image_embed = model.image_encoder(inputs['pixel_values'])[0][0, :, :]
print(image_embed.shape)
print(image_embed[0][0])


image_features = model.clip.image_encode(inputs['pixel_values'])
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features = model.clip.text_encode(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
text_features /= text_features.norm(dim=-1, keepdim=True)

logits = np.exp(4.6052) * image_features @ text_features.T
probs = logits.softmax(dim=1)
print(probs)


print('----partial----')
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
image_features = model(inputs['pixel_values']).image_embeds
image_features /= image_features.norm(dim=-1, keepdim=True)

logits = np.exp(4.6052) * image_features @ text_features.T
print(logits)
probs_ = logits.softmax(dim=1)
print(probs_)


print('----ImaneNetV2 zero-shot----')
device = 'cuda'
model, tokenizer, transform = factory.create()
model = model.to(device)
dataset = ImageNetV2Dataset(transform=transform('valid')) 
evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
eval_stats = evaluator(model.clip)
print(eval_stats)

