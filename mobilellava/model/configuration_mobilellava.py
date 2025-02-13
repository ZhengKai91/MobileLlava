from transformers import PretrainedConfig
from dataclasses import dataclass, asdict, fields
import logging

class MobileLlavaConfig(PretrainedConfig):
    model_type = "mobilellava"
    def __init__(
            self,
            vision_name_or_path=None,
            llm_name_or_path=None,
            vision_config={},
            llm_config={},
            tokenizer_padding_side='right',
            max_seq_length=2048,
            tokenizer_use_fast=False,
            vision_feature_layer=-2,
            img_context_token_id=None,
            **kwargs):
        super().__init__(**kwargs)

        self.vision_name_or_path = vision_name_or_path
        self.llm_name_or_path = llm_name_or_path
        self.vision_config = vision_config
        self.llm_config = llm_config
        self.tokenizer_padding_side=tokenizer_padding_side
        self.max_seq_length = max_seq_length
        self.tokenizer_use_fast = tokenizer_use_fast
        self.vision_feature_layer = vision_feature_layer
        self.img_context_token_id = img_context_token_id
    
    def update(self, model_args):
        for name in vars(model_args):
            if hasattr(self, name):
                val = getattr(model_args, name)
                setattr(self, name, val )
    
    def print(self):
        base_instance = PretrainedConfig()
        base_attrs = vars(base_instance).keys()
        instance_attrs = vars(self)
        own_attrs = {k: v for k, v in instance_attrs.items() if k not in base_attrs and not k.startswith('_')}
        logging.info('MobileLlavaConfig:')
        for attr, value in own_attrs.items():
            logging.info(f"{attr}: {value}")
