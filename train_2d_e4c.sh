MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="models/colorpeel_e4c_maroon2olive_10000steps_multimap"

CUDA_VISIBLE_DEVICES=2 python ./train_e4t.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list=./concept_json/instance_2s4c_rgby.json \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --pre_step=1000 \
  --checkpointing_steps 1000 \
  --scale_lr --hflip  \
  # --modifier_token "<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>" \
  # --initializer_token "rectangle+circle+red+green+blue+yellow"

# CUDA_VISIBLE_DEVICES=2 python ./test.py --exp colorpeel_2d_2s4c
