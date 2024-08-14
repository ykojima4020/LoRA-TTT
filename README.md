# MAE-CLIP
Abstract here.

## Overview
This repository provides the following features to replicate our method:
* Applying LoRA into OpenAI CLIP
* MAE fine-tuning
* Single Instance TTA
* Evaluation on zero-shot image classification

## Setup
---
### Prerequisites
* Ubuntu 23.10
* Python 3.11
* CUDA (recommend)

### install
1. `git clone --recurse-submodules https://github.com/ykojima4020/mae_clip.git`
2. `git submodule update --init --recursive`
3. `cd mae_clip`
4. `virtualenv venv --python=python3`
5. `source venv/bin/activate`
6. `pip install -r requirements.txt`

### Datasets

#### Fine-tuning
> [!NOTE] 
> If you want to perform TTA without fine-tuning, you don't need to download this dataset.

We use [webdataset](https://webdataset.github.io/webdataset/) for scalable data loading during fine-tuning based on the implementation of [GroupViT](https://github.com/NVlabs/GroupViT?tab=readme-ov-file#data-preparation).
> To convert image text pairs into the webdataset format, we use the [img2dataset](https://github.com/rom1504/img2dataset) tool to download and preprocess the dataset.

We use just 5% of the CC3M dataset during fine-tuning. After downloading the entire dataset by following [this page](https://github.com/NVlabs/GroupViT?tab=readme-ov-file#gcc3m), select the following tar files to constitute this 5%.
<details>
<summary>files</summary>

```txt
gcc-train-000218.tar
gcc-train-000138.tar
gcc-train-000148.tar
gcc-train-000048.tar
gcc-train-000109.tar
gcc-train-000078.tar
gcc-train-000253.tar
gcc-train-000318.tar
gcc-train-000176.tar
gcc-train-000151.tar
gcc-train-000212.tar
gcc-train-000056.tar
gcc-train-000113.tar
gcc-train-000029.tar
gcc-train-000107.tar
gcc-train-000132.tar
```
</details>

You can add this dataset to the configuration file as shown below. The path and exact number of files in the dataset will depend on your environment.
```yaml
data:
  dataset:
    meta:
      gcc3m_005:
        type: img_txt_pair
        path: '/path/to/dataset'
        prefix: gcc-train-{000000..001242}.tar
        length: 112810
    train: ['gcc3m_005']
```

#### TTA
Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets) to download the dataset that you want to evaluate. Additionally, please adjust `path` for each dataset in the configuration YAML file to match the directories where you downloaded them.

```yaml
data:
  dataset:
    meta:
      imagenet:
        type: img_classification
        path: '/path/to/dataset'
        classes: 'imagenet'
        prompt: 'ensemble'
```

## Run
`python multi_gpu_mae_clip_runner.py --cfg config/mae_clip_run.yaml`  
You can run it with various settings by editing the configuration file.

### TTA dataset
You can change the dataset used for evaluating TTA with  `data.dataset.tta` in the configuration YAML file.

```yaml
data:
  dataset:
    tta: ['imagenet', 'imagenet_a', 'imagenetv2', 'imagenet_r', 'imagenet_sketch']
```

### Text Prompts
`simple` means 'a photo of a {class text}'.  
`ensemble` refers to the average of 80 templates proposed in the CLIP paper.  
For more details, check [here](./evaluator/imagenet_config.py)

```
 prompt: 'ensemble' or 'simple'
```

### Target parameters and Test-Time loss
You can specify the target parameters in the `tta.params` key using either `['peft']` or `['tp']`. `'peft'` indicates that TTA updates the LoRA parameters in the image encoder, while `'tp'` refers to text prompts similar to TPT. Test-Time losses are specified under each key using the `loss` key, which can be set to either `['mem']`, `['mae']`, or both `['mae', 'mem']`. When using `'tp'`, only the `['mem']` loss is supported.

```yaml
tta:
  peft:
    batch_size: 64
    epochs: 1
    optimizer: 'adam'
    lr: 1e-3
    weight_decay: 0.2
    eps: 1e-8
    betas: [0.9, 0.95]
    loss: ['mem'] or ['mae'] or ['mae', 'mem']
    mae:
      weight: 1
    mem:
      weight: 1

  tp:
    lr: 5e-3
    batch_size: 64
    epochs: 1
    loss: ['mem']
    mem:
      weight: 1

  params: ['peft'] or ['tp']
  run_freq: 1
```

### Fine-tuning and checkpoint
When the `finetune` key is set to `true`, fine-tuning will be performed before TTA. Additionally, you can specify the checkpoint path to start evaluation from specific weights of LoRA and the MAE decoder.

```yaml
finetune: true or false
checkpoint: false or checkpoint path
```

### Benchmark Results
#### Out-of-Distribution Generalization
To be released soon.
#### Cross-Dataset Generalization
To be released soon.

## References
This repository is implemented based on the following references.
* https://github.com/moein-shariatnia/OpenAI-CLIP
* https://github.com/azshue/TPT
* https://github.com/NVlabs/GroupViT
