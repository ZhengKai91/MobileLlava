from dataclasses  import  dataclass
import numpy as np
import torch
from .constants import IMG_CONTEXT_TOKEN, IGNORE_TOKEN_ID

@dataclass
class ConvTemplate:
    name: str="default"
    system_template:str='{system_message} '
    system_message:str=''
    user_template:str='USER: {content} '
    gpt_template:str='ASSISTANT: {content}<|endoftext|>'
    assistant_ignore:str="ASSISTANT: "
    image_token_first:bool=True
    num_image_tokens:int=1

    def __call__(self, conversations, tokenizer):
        new_conversations = conversations[1:] if conversations[0]['from'] != 'human' else conversations 
        system_prompt = self.system_template.format(system_message=self.system_message)
        prompts = [{'role':'system', 'content':system_prompt}]
        for conversation in new_conversations:
            role = conversation['from']
            if role =='human':
                placeholder = '<image>'
                value = conversation['value'] if self.name !='pretrain' else placeholder
                text_only = value.count(placeholder) < 1
                if not text_only: 
                    image_tokens = IMG_CONTEXT_TOKEN * self.num_image_tokens
                    value = self.format_image_placeholder(value, placeholder, image_tokens, self.image_token_first )
                value = self.user_template.format(content=value)
            elif role =='gpt':
                value = self.gpt_template.format(content=conversation['value'])
            else:
                raise NotImplementedError
            prompts.append({'role':role, 'content':value})
        ret = self.make_inputs_labels(prompts, tokenizer)
        #print(prompts)
        return ret

    def make_inputs_labels(self, prompts, tokenizer):
        messages = [prompt['content'] for prompt in prompts]
        roles = [prompt['role'] for prompt in prompts]
        #print(messages)
        input_ids = tokenizer(messages, 
            add_special_tokens=False, 
            return_tensors='np', 
            padding=False, 
            max_length=tokenizer.model_max_length, 
            truncation=False).input_ids

        final_input_ids, final_targets = [], []
        ignore_ids = tokenizer(self.assistant_ignore, add_special_tokens=False, return_tensors='np').input_ids[0]
        ignore_len = ignore_ids.shape[0]
        for role, input_id in zip(roles, input_ids):
            final_input_ids.append(input_id)
            if role =='human' or role=='system':
                final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))
            else:
                target = input_id.copy()
                target[:ignore_len] = IGNORE_TOKEN_ID
                final_targets.append(target)
        input_ids = torch.tensor(np.concatenate(final_input_ids), dtype=torch.long)
        targets = torch.tensor(np.concatenate(final_targets),dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels = targets,
            attention_mask = input_ids.ne(tokenizer.pad_token_id) #position_ids?
        )

    def format_image_placeholder(self, value, placeholder, image_tokens, image_token_first):
        if image_token_first:
            # format : <image>\n {content w/o <image>}
            image_cnt = value.count(placeholder)
            assert image_cnt == 1, f'only support 1 image for now. {image_cnt}'
            value= placeholder + "\n" +  value.replace(placeholder, '').strip()
        value = value.replace(placeholder, image_tokens).strip()
        return value
 
def get_conv_template(name, num_image_tokens=1):
    if name == 'pretrain':
        return ConvTemplate(
                    name='pretrain', 
                    system_template ='{system_message}',
                    system_message="",
                    user_template ="{content}",
                    gpt_template ='{content}\n',
                    assistant_ignore="",
                    num_image_tokens = num_image_tokens
                )
    elif name == 'qwen2_base':
        return ConvTemplate(
                    name='qwen2_base', 
                    system_template ='<|im_start|>system\n{system_message}<|im_end|>\n',
                    system_message="You are a helpful assistant. ",
                    user_template ='<|im_start|>user\n{content}<|im_end|>\n',
                    gpt_template = '<|im_start|>assistant\n{content}<|im_end|><|endoftext|>\n', 
                    assistant_ignore="<|im_start|>assistant\n",
                    num_image_tokens = num_image_tokens
                )
    elif name == 'mobilellm':
        return ConvTemplate(
                    name='mobilellm', 
                    system_template ='{system_message}\n',
                    system_message="You are a helpful assistant.",
                    user_template ='USER:\n{content}\n',
                    gpt_template = 'ASSISTANT:\n{content}</s>', 
                    assistant_ignore="ASSISTANT:\n",
                    num_image_tokens = num_image_tokens
                )
    else:
        raise NotImplementedError
