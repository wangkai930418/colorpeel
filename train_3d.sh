MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="models/colorpeel_2d_2s4c"

CUDA_VISIBLE_DEVICES=2 python ./train_colorpeel.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list=./concept_json/instance_2s4c_rgby.json \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --scale_lr --hflip  \
  --modifier_token "<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>" \
  --initializer_token "rectangle+circle+red+green+blue+yellow"

CUDA_VISIBLE_DEVICES=2 python ./test.py --exp colorpeel_2d_2s4c

  # --modifier_token "<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>" \
  # --initializer_token "rectangle+circle+red+green+blue+yellow"

  # exp7 - 251.5,3.5,3.5 --- best model
  # exp8 - 235,235,33
  # exp9 - w2c mean threshold 0.8 
  # exp10 - w2c mean threshold 0.8

  # colorpeel_exp10_2d_2s4c_w2c_cos_0.2_3000_100_42_6.0_10_personlized best model 2D
