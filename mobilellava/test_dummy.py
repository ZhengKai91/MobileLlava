from train.dataset import LazySupervisedDataset, PadDataCollator
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer,CLIPImageProcessor
from train.constants import IMG_CONTEXT_TOKEN, IGNORE_TOKEN_ID
from model.modeling_mobilellava import MobileLlavaChatModel
from model.configuration_mobilellava import MobileLlavaConfig


if __name__=="__main__":
    data_json = '/Users/kaizheng/Work/projects/dataset/sft_data.json' 

    config = MobileLlavaConfig()
    config.vision_name_or_path = '/Users/kaizheng/Work/models/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'
    config.llm_name_or_path = '/Users/kaizheng/Work/models/MobileLLM-125M'#QWen2-0.5B
    model = MobileLlavaChatModel(config)
    tokenizer = model.tokenizer
    image_processor = CLIPImageProcessor.from_pretrained(config.vision_name_or_path)
    
    num_new_tokens = tokenizer.add_tokens([IMG_CONTEXT_TOKEN], special_tokens=True)

    train_dataset = LazySupervisedDataset(data_json, tokenizer, 'mobilellm', image_processor, 1)
    collate_fn = PadDataCollator(tokenizer)
    
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=1, collate_fn=collate_fn)  
    for batch in dataloader:
        import pdb; pdb.set_trace()
        print('batch')
