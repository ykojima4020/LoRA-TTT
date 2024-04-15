import os.path as osp

import torch
import torch.distributed as dist

from data.dataset import CLIPDataset
from misc.coco_captions_to_df import get_coco_captions_df, get_coco_captions_test_df

from braceexpand import braceexpand
import webdataset as wds

class CLIPDataLoaderBuilder():

    def __init__(self, cfg, tokenizer, transform):
        self._tokenizer = tokenizer
        self._batch_size = cfg.batch_size
        self._num_workers = cfg.num_workers
        self._pin_memory = cfg.pin_memory
        self._transform = transform

    def __call__(self, image_path, annotation_json, mode, rank, world_size, test=False):
        if not test:
            dataframe = get_coco_captions_df(annotation_json)
        else:
            dataframe = get_coco_captions_test_df(annotation_json)

        dataset = CLIPDataset(
            image_path,
            dataframe["image"].values,
            dataframe["caption"].values,
            tokenizer=self._tokenizer,
            transforms=self._transform(mode),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=dist.get_rank(),
            shuffle=True if mode == "train" else False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, num_workers=self._num_workers,
            pin_memory = self._pin_memory,
            shuffle=sampler is None,
            sampler=sampler)
        return dataloader, sampler

class GCC3MDataLoaderBuilder():

    def __init__(self, cfg, tokenizer, transform):
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._batch_size = cfg.batch_size
        self._num_workers = cfg.num_workers
        self._pin_memory = cfg.pin_memory
        self._max_length = cfg.text_aug.max_seq_len
        self._shuffle_buffer = cfg.shuffle_buffer
        self._distributed = True
        self._transform = transform

    def __call__(self, mode, rank, world_size, test=False):
        dataset_type = None
        tar_file_list = []
        total_length = 0
        for ds in self._cfg.dataset[mode]:
            ds_meta = self._cfg.dataset.meta[ds]
            if dataset_type is None:
                dataset_type = ds_meta.type
            else:
                assert dataset_type == ds_meta.type, \
                    'All datasets must be of the same type'
    
            prefix = ds_meta.prefix
            path = ds_meta.path
            length = ds_meta.length
            cur_tar_file_list = []
            for tar_file in braceexpand(osp.join(path, prefix)):
                if osp.exists(tar_file):
                    cur_tar_file_list.append(tar_file)
            print(f'Found {len(cur_tar_file_list)} files for dataset {ds}')
            tar_file_list.extend(cur_tar_file_list)
            total_length += length
        print(f'Found {len(tar_file_list)} files in total for split {mode}')

        dataset = wds.WebDataset(tar_file_list, shardshuffle=True, nodesplitter=wds.split_by_node)
        dataset = dataset.shuffle(self._shuffle_buffer)
        dataset = dataset.decode('pil')
        dataset = dataset.rename(image='jpg;png;jpeg', text='text;txt', keep=False)
        dataset = dataset.map_dict(image=self._transform(mode), text=self._text_transform)
        # dataset = dataset.with_length(total_length)

        dataloader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)
        dataloader = dataloader.shuffle(7) # shuffle across the loader workers
        if self._distributed:
            dataloader = dataloader.batched(batchsize=self._batch_size, collation_fn=self._collate_cb, partial=False)
        else:
            dataloader = dataloader.batched(batchsize=self._batch_size, collation_fn=self._collate_cb)

        # [NOTE] The following codes are from https://github.com/tmbdev-archive/webdataset-examples/blob/master/main-wds.py#L324-L335
        if self._distributed:
            # With DDP, we need to make sure that all nodes get the same number of batches;
            # we do that by reusing a little bit of data.
            # Note that you only need to do this when retrofitting code that depends on
            # epoch size. A better way is to iterate through the entire dataset on all nodes.
            num_batches = max(1, total_length // (self._batch_size * world_size))
            print("# batches per node = ", num_batches)
            dataloader.length = num_batches
            dataloader = dataloader.with_length(num_batches)

            dataloader = dataloader.repeat(nbatches=num_batches)
            dataloader = dataloader.slice(num_batches)
            # This only sets the value returned by the len() function; nothing else uses it,
            # but some frameworks care about it.
        return dataloader, None

    def _collate_cb(self, batch):
        images = list()
        input_ids = list()
        attention_masks = list()
    
        for b in batch:
            images.append(b['image'])
            input_ids.append(b['text']['input_ids'][0])
            attention_masks.append(b['text']['attention_mask'][0])

        images = torch.stack(images, dim=0)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        return {'image': images, 'input_ids': input_ids, 'attention_mask': attention_masks} 

    def _text_transform(self, x):
        # [NOTE]: truncation cut the sequence in a fixed size.
        encode = self._tokenizer(x, padding="max_length", truncation=True, max_length=self._max_length)
        return encode
