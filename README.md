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
gcc-train-000000.tar
gcc-train-000002.tar
gcc-train-000008.tar
gcc-train-000009.tar
gcc-train-000010.tar
gcc-train-000011.tar
gcc-train-000012.tar
gcc-train-000018.tar
gcc-train-000019.tar
gcc-train-000020.tar
gcc-train-000021.tar
gcc-train-000023.tar
gcc-train-000029.tar
gcc-train-000030.tar
gcc-train-000031.tar
gcc-train-000033.tar
gcc-train-000037.tar
gcc-train-000040.tar
gcc-train-000041.tar
gcc-train-000044.tar
gcc-train-000045.tar
gcc-train-000048.tar
gcc-train-000049.tar
gcc-train-000050.tar
gcc-train-000051.tar
gcc-train-000052.tar
gcc-train-000055.tar
gcc-train-000056.tar
gcc-train-000061.tar
gcc-train-000066.tar
gcc-train-000068.tar
gcc-train-000069.tar
gcc-train-000075.tar
gcc-train-000076.tar
gcc-train-000077.tar
gcc-train-000078.tar
gcc-train-000080.tar
gcc-train-000082.tar
gcc-train-000083.tar
gcc-train-000085.tar
gcc-train-000087.tar
gcc-train-000089.tar
gcc-train-000090.tar
gcc-train-000091.tar
gcc-train-000092.tar
gcc-train-000093.tar
gcc-train-000094.tar
gcc-train-000095.tar
gcc-train-000098.tar
gcc-train-000100.tar
gcc-train-000107.tar
gcc-train-000108.tar
gcc-train-000109.tar
gcc-train-000111.tar
gcc-train-000113.tar
gcc-train-000118.tar
gcc-train-000120.tar
gcc-train-000121.tar
gcc-train-000122.tar
gcc-train-000123.tar
gcc-train-000124.tar
gcc-train-000125.tar
gcc-train-000128.tar
gcc-train-000130.tar
gcc-train-000131.tar
gcc-train-000132.tar
gcc-train-000134.tar
gcc-train-000137.tar
gcc-train-000138.tar
gcc-train-000139.tar
gcc-train-000141.tar
gcc-train-000145.tar
gcc-train-000147.tar
gcc-train-000148.tar
gcc-train-000150.tar
gcc-train-000151.tar
gcc-train-000152.tar
gcc-train-000155.tar
gcc-train-000156.tar
gcc-train-000157.tar
gcc-train-000158.tar
gcc-train-000161.tar
gcc-train-000162.tar
gcc-train-000163.tar
gcc-train-000170.tar
gcc-train-000172.tar
gcc-train-000173.tar
gcc-train-000174.tar
gcc-train-000175.tar
gcc-train-000176.tar
gcc-train-000177.tar
gcc-train-000179.tar
gcc-train-000180.tar
gcc-train-000182.tar
gcc-train-000184.tar
gcc-train-000187.tar
gcc-train-000189.tar
gcc-train-000191.tar
gcc-train-000192.tar
gcc-train-000193.tar
gcc-train-000195.tar
gcc-train-000196.tar
gcc-train-000201.tar
gcc-train-000203.tar
gcc-train-000207.tar
gcc-train-000212.tar
gcc-train-000214.tar
gcc-train-000215.tar
gcc-train-000217.tar
gcc-train-000218.tar
gcc-train-000220.tar
gcc-train-000223.tar
gcc-train-000227.tar
gcc-train-000228.tar
gcc-train-000230.tar
gcc-train-000231.tar
gcc-train-000241.tar
gcc-train-000243.tar
gcc-train-000244.tar
gcc-train-000247.tar
gcc-train-000248.tar
gcc-train-000250.tar
gcc-train-000252.tar
gcc-train-000253.tar
gcc-train-000254.tar
gcc-train-000262.tar
gcc-train-000263.tar
gcc-train-000264.tar
gcc-train-000265.tar
gcc-train-000267.tar
gcc-train-000270.tar
gcc-train-000272.tar
gcc-train-000274.tar
gcc-train-000276.tar
gcc-train-000277.tar
gcc-train-000280.tar
gcc-train-000281.tar
gcc-train-000283.tar
gcc-train-000285.tar
gcc-train-000286.tar
gcc-train-000287.tar
gcc-train-000288.tar
gcc-train-000290.tar
gcc-train-000293.tar
gcc-train-000294.tar
gcc-train-000296.tar
gcc-train-000297.tar
gcc-train-000299.tar
gcc-train-000301.tar
gcc-train-000302.tar
gcc-train-000303.tar
gcc-train-000304.tar
gcc-train-000306.tar
gcc-train-000307.tar
gcc-train-000309.tar
gcc-train-000311.tar
gcc-train-000314.tar
gcc-train-000315.tar
gcc-train-000318.tar
gcc-train-000320.tar
gcc-train-000322.tar
gcc-train-000324.tar
gcc-train-000325.tar
gcc-train-000327.tar
gcc-train-000328.tar
gcc-train-000331.tar
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
