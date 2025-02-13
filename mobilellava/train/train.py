import os
import json
import logging
from dataclasses import dataclass, asdict
from transformers import HfArgumentParser, TrainingArguments, Trainer

from mobilellava.model.configuration_mobilellava import MobileLlavaConfig
from mobilellava.model.modeling_mobilellava import MobileLlavaChatModel
from mobilellava.utils.dist_utils import init_dist 
from .dataset import LazySupervisedDataset, PadDataCollator
from .constants import  IMG_CONTEXT_TOKEN

from transformers import CLIPProcessor, CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
#import torch_npu

logging.basicConfig(level=logging.INFO)

@dataclass
class DataArguments:
    data_json: str=None
    template_name: str='pretrain'

@dataclass
class ModelArguments:
    model_name_or_path: str=None
    vision_name_or_path: str=None
    llm_name_or_path: str=None
    trust_remote_code:bool=True
    attn_implementation:str='flash_attention_2'
    freeze_vision: bool=False
    freeze_connector: bool=False
    freeze_llm: bool=False
    max_seq_length:int=2048
    vision_feature_layer:int=-2
    tokenizer_padding_side:str='right'
    tokenizer_use_fast:bool=False
    special_tokens:str='{"additional_special_tokens":[]}'
    backend:str='hccl'


def train():
    launcher = os.environ.get('LAUNCHER', 'pytorch')

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    possible_device = init_dist(launcher=launcher, backend=model_args.backend)
    torch_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if model_args.model_name_or_path is not None:
        logging.info('load from config')
        config = MobileLlavaConfig.from_pretrained(model_args.model_name_or_path, attn_implementation=model_args.attn_implementation)
        model = MobileLlavaChatModel.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, config=config)
    else:
        assert model_args.vision_name_or_path is not None and model_args.llm_name_or_path is not None
        vision_model = CLIPVisionModel.from_pretrained(
                model_args.vision_name_or_path, 
                trust_remote_code=model_args.trust_remote_code,
                #attn_implementation = model_args.attn_implementation
            )
        llm_model = AutoModelForCausalLM.from_pretrained(
                model_args.llm_name_or_path, 
                trust_remote_code=model_args.trust_remote_code,
                attn_implementation = model_args.attn_implementation)
        config = MobileLlavaConfig() 
        config.update(model_args)
        kwargs = {'trust_remote_code':model_args.trust_remote_code}
        model = MobileLlavaChatModel(config, vision_model, llm_model, **kwargs)
        model.to(torch_dtype)
    config.print()

    if model.img_context_token_id is None:
        special_tokens = json.loads(model_args.special_tokens)
        additional_special_tokens = special_tokens.get('additional_special_tokens', [])
        if IMG_CONTEXT_TOKEN not in additional_special_tokens:
            additional_special_tokens.append(IMG_CONTEXT_TOKEN)
        special_tokens['additional_special_tokens']= additional_special_tokens
        logging.info(f'specail_tokens:{special_tokens}')
        model.add_special_tokens(special_tokens)
        model.img_context_token_id = model.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    model.freeze_params(model_args.freeze_vision, model_args.freeze_connector, model_args.freeze_llm)
    model.gradient_checkpointing_enable() 
    model.llm_model._set_gradient_checkpointing()

    device = torch.device(possible_device)
    model.to(device)

    image_processor = CLIPImageProcessor.from_pretrained(config.vision_name_or_path)
    train_dataset = LazySupervisedDataset(data_args.data_json, model.tokenizer, data_args.template_name, image_processor, model.num_image_tokens)
    collator = PadDataCollator(model.tokenizer)
    
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=None, tokenizer=model.tokenizer, data_collator=collator)
    train_result = trainer.train()
    trainer.save_model()

if __name__=="__main__":
    train()
