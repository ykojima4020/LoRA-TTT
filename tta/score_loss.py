def get_mean_score_and_loss(model, data_loader, text_embeddings, device):
    # [NOTE]: initial calculation
    logit_scale = model.clip.logit_scale
    mae_loss_meter = AvgMeter()
    clip_score_meter = AvgMeter()
    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            mae_loss, reconstruction, mask = model.mae(images)
            count = images.size(0)
            mae_loss_meter.update(mae_loss.item(), count)

            clip_score = get_score(model, text_embeddings, images, targets, logit_scale=logit_scale.exp())
            clip_score_meter.update(clip_score.item(), count)

    return clip_score_meter.avg, mae_loss_meter.avg
 
def get_score(model, zeroshot_weights, images, targets, logit_scale=100):
    with torch.no_grad():
        image_features = model.clip.image_encode(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = (logit_scale * image_features @ zeroshot_weights).softmax(dim=-1)

    # [NOTE]: this is tricky. validaty is already comfirmed.
    targets_expand = torch.unsqueeze(targets, 0)
    n = scores.shape[0]
    results = scores[np.arange(n), targets_expand].squeeze(0)
    return torch.mean(results)
 

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

