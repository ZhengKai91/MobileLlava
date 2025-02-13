# What is this?

MobileLlava is a simple and easy-to-learn framework for training large multimodal models. It allows you to start training multimodal models from scratch directly on a Mac. It features:
Super Lightweight: You can start training with just one Mac.
Easy to Expand: With only a few lines of configuration changes, you can train various multimodal models combining different encoders and LLMs.
Reliable Training Results: It is not just a toy demo but delivers reliable performance.

1. **Super Light**: You can even start training with just one Mac.
2. **Easy to Expand**: With only a few lines of configuration changes, you can train various multimodal models combining different vision encoders and LLMs.
3. **Reliable Training Results**:It is not just a toy demo but delivers reliable performance.

# Training
## Pretrain
```sh
./scripts/pretrain.sh
```
Modify the following parameters according to your environment：
```python
#torchrun --nnodes=1 --node_rank=0  --master_addr=127.0.0.1 --nproc_per_node=1 --master_port=34229 \
python \
-m mobilellava.train.train \
  --output_dir outputs\
  --vision_name_or_path wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M \ #vision encder 
  --llm_name_or_path facebook/MobileLLM-125M \ #llm path
  --data_json ./data/pretrain_data.json \ #your data json 
  --template_name 'pretrain'\
  --attn_implementation sdpa \ # eager / sdpa / flash_attention_2...
  --special_tokens '{"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>"}' \
  --backend gloo\ # nccl /gloo / hccl
  --dataloader_num_workers 8 \
  --learning_rate 1e-3 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --bf16 True \
  --max_seq_length 2048 \
  --logging_steps 1 \
```

## SFT
```sh
./scripts/finetune.sh
```

# Dataset
The dataset used in the training script is a subset of LLaVA 1.5 and is intended for demonstration purposes only. For full training, please download the complete dataset.