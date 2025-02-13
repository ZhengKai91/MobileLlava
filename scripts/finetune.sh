#torchrun --nnodes=1 --node_rank=0  --master_addr=127.0.0.1 --nproc_per_node=1 --master_port=34229 \
python -m \
mobilellava.train.train \
  --output_dir outputs/finetune\
  --model_name_or_path outputs/ \
  --data_json /Users/kaizheng/Work/projects/MobileLlava/data/sft_data.json \
  --template_name mobilellm\
  --attn_implementation sdpa \
  --backend gloo \
  --dataloader_num_workers 1 \
  --freeze_vision True\
  --learning_rate 1e-3 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --bf16 False \
  --max_seq_length 2048 \
  --logging_steps 1 \
  --do_train True \
