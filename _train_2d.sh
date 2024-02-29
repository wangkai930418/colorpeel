# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="models/colorpeel_2d_2s4c4p"

# CUDA_VISIBLE_DEVICES=7 python ./train_colorpeel.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --output_dir=$OUTPUT_DIR \
#   --concepts_list=./concept_json/colorpeel_instances_2d.json \
#   --resolution=512  \
#   --train_batch_size=1  \
#   --learning_rate=1e-5  \
#   --lr_warmup_steps=0 \
#   --max_train_steps=3000 \
#   --scale_lr --hflip  \
#   --modifier_token "<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>" \
#   --initializer_token "rectangle+circle+red+green+blue+yellow"

CUDA_VISIBLE_DEVICES=7 python ./test.py --samples 13 --exp colorpeel_2d_2s4c4p
