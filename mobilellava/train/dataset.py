import transformers
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import copy
import os
from PIL import Image
from .conv_template import get_conv_template
from .constants import IGNORE_TOKEN_ID

class LazySupervisedDataset(Dataset):
    def __init__(self, data_json, tokenizer, template_name, image_processor, num_image_tokens):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = self.load_data_dict(data_json)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.template = get_conv_template(template_name, num_image_tokens)

    def load_data_dict(self, data_json):
        list_data_dict = []
        with open(data_json, 'r') as f:
            datasets= json.load(f)
        for name in datasets:
            anno_file = datasets[name]['annotation']
            items = json.load(open(anno_file, 'r'))
            for item in items:
                item['image_root'] = datasets[name]['image_root']
            list_data_dict.extend(items)
        print(f'load {len(list_data_dict)} image-text pair')
        return list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        data_item = self.list_data_dict[i]
        conversations = copy.deepcopy(data_item['conversations'])
        ret = self.template(conversations, self.tokenizer) # ret :{'input_ids':xxx, 'labels':xxx, 'attention_mask'}
        if 'image' in data_item:
            image_path = os.path.join(data_item['image_root'], data_item['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
            image_flags=torch.tensor([1], dtype=torch.long)
        else:
            crop_size = self.image_processor.crop_size
            image = torch.zeros(3, crop_size['height'], crop_size['width'])
            image_flags=torch.tensor([0], dtype=torch.long)
        ret['images'] = image
        ret['image_flags'] = image_flags

        return ret

class PadDataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [fea['input_ids'] for fea in features]
        labels = [fea['labels'] for fea in features]
        attention_mask = [fea['attention_mask'] for fea in features] 
        images = [fea['images'] for fea in features]
        image_flags = [fea['image_flags'] for fea in features]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        images = torch.stack(images)
        image_flags = torch.stack(image_flags)
        batch = dict(
                input_ids=input_ids, 
                labels=labels, 
                attention_mask=attention_mask, 
                images=images,
                image_flags = image_flags
            )
        return batch

