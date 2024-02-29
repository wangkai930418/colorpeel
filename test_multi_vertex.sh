for OBJECT in   'chair' ### 'ball' 'bowl' 'plate' 'teddybear' 'sofa' 'mug' 'horse' 
# for OBJECT in   'chair' ### 'ball' 'bowl' 'plate' 'vase' 'pants' 'teddybear' \
                    ###'sofa' 'rose' 'parrot' 'horse' 'beverage' 'fountain' 'mug' \

do

CUDA_VISIBLE_DEVICES=3 python test_e4c_multi_vertex.py \
    --object "${OBJECT}" \
    --exp "colorpeel_e4c_sienna2indigo_10000steps_multi_vertex" \
    --inf_steps 25 \

done
