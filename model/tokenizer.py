import torch

class BertTokenizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts, padding=True, truncation=True, max_length=None):
        # [NOTE]: when texts is str, this tokenizer returns a single list,
        #         but, when texts is list, this torkenizer returns a multiple list.
        if isinstance(texts, str):
            texts = [texts] 
        tokens = self.tokenizer(texts, padding=padding, truncation=truncation, max_length=max_length)
        return {'input_ids': torch.tensor(tokens['input_ids']),
                'attention_mask': torch.tensor(tokens['attention_mask'])}

class OpenCLIPTokenizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts, padding=True, truncation=True, max_length=None):
        # [NOTE]: the parameters above are dummy
        tokens = self.tokenizer(texts)
        # [NOTE]: attention_mask is also dummy
        return {'input_ids': tokens, 'attention_mask': torch.zeros(tokens.shape)}

if __name__ == "__main__":
    from transformers import DistilBertTokenizer
    import open_clip

    import argparse
    from misc.config import load_config
    from omegaconf import OmegaConf

    from data.dataloader_builder import CLIPDataLoaderBuilder

    parser = argparse.ArgumentParser('Evaluation on ImageNet-C', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)    

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bertokenizer = BertTokenizer(tokenizer)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    opentokenizer = OpenCLIPTokenizer(tokenizer)

    inputs = ['hoge', 'fuga fuga']
    tokens = bertokenizer(inputs, max_length=1)
    print(tokens)

    tokens = opentokenizer(inputs, max_length=1)
    print(tokens)


    dataloader_builder = CLIPDataLoaderBuilder(cfg.data, bertokenizer)

    val_loader, _ = dataloader_builder(cfg.data.dataset.val_image_path,
                                      cfg.data.dataset.val_json, 'val', 0, 1, test=cfg.test)
