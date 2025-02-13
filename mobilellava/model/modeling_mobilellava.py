import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
import logging
from transformers.modeling_utils import PreTrainedModel, GenerationConfig
from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_mobilellava import MobileLlavaConfig

class Connector(nn.Module):
    def __init__(self, vit_hidden_size, llm_hidden_size):
        super().__init__()
        self._connector = nn.Sequential(
                #nn.LayerNorm(vit_hidden_size),
                nn.Linear(vit_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
             )
    def forward(self, x):
        return self._connector(x)

class MobileLlavaChatModel(PreTrainedModel):
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    def __init__(self, config, vision_model=None, llm_model=None, **kwargs):
        super().__init__(config)
        trust_remote_code = kwargs.get('trust_remote_code', True)
        if vision_model is not None: 
            self.vision_model = vision_model
        else:
            vision_config = AutoConfig.from_pretrained(
                    config.vision_name_or_path, 
                    trust_remote_code=trust_remote_code).vision_config
            #vision_config._attn_implementation = self.config._attn_implementation
            if self.config.vision_config:
                vision_config = self.update_config(vision_config, self.config.vision_config)
            self.vision_model = CLIPVisionModel(vision_config)

        if llm_model is not None: 
            self.llm_model = llm_model
        else:
            llm_config = AutoConfig.from_pretrained(
                    config.llm_name_or_path, 
                    trust_remote_code=trust_remote_code, 
                    attn_implementation=self.config._attn_implementation)
            if self.config.llm_config:
                llm_config = self.update_config(llm_config, self.config.llm_config) 
            self.llm_model = AutoModelForCausalLM.from_config(llm_config, trust_remote_code=trust_remote_code)

        self.connector = Connector(self.vision_model.config.hidden_size, self.llm_model.config.hidden_size)
        tokenizer_path = config._name_or_path or self.config.llm_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                model_max_length = config.max_seq_length,
                padding_side = config.tokenizer_padding_side,
                use_fast = config.tokenizer_use_fast,
                trust_remote_code=trust_remote_code
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        self.img_context_token_id = self.config.img_context_token_id
        image_size = self.vision_model.config.image_size
        patch_size = self.vision_model.config.patch_size
        self.num_image_tokens =int((image_size // patch_size) ** 2)

    def add_special_tokens(self, tokens):
        num_new_tokens = self.tokenizer.add_special_tokens(tokens)
        if num_new_tokens > 0:
            logging.info(f'add {num_new_tokens} new special tokens')
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            self.llm_model.config.vocab_size = len(self.tokenizer)
            self.config.llm_config.update({'vocab_size':self.llm_model.config.vocab_size})
    
    def extract_vision_feature(self, pixel_values):
        vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            ).hidden_states[self.config.vision_feature_layer]
        vit_embeds = vit_embeds[:, 1:]
        return vit_embeds

    def forward(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        image_flags = image_flags.squeeze(-1)
        vit_embeds = self.extract_vision_feature(images)
        img_tokens = self.connector(vit_embeds)
        img_tokens = img_tokens[image_flags==1]

        input_embeds = self.llm_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        input_embeds[selected] = input_embeds[selected] * 0.0 + img_tokens.reshape(-1, C)
       
        ignore_flag = False
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.llm_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.llm_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        input_embeds = self.llm_model.get_input_embeddings()(input_ids)
        if images is not None:
            vit_embeds = self.extract_vision_feature(images)
            img_tokens = self.connector(vit_embeds)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = img_tokens.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        outputs = self.llm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs


    def update_config(self, config, input_dict):
        for key in input_dict.keys():
            if hasattr(config, key):
                setattr(config, key, input_dict[key])
        return config

    def freeze_params(self, freeze_vision=False, freeze_connector=False, freeze_llm=False):
        if freeze_vision:
            self._freeze_module(self.vision_model)
            logging.info('freeze vision_model')
        if freeze_connector:
            self._freeze_module(self.connector)
            logging.info('freeze connector')
        if freeze_llm:
            self._freeze_module(self.llm_model)
            logging.info('freeze llm_model')

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        
