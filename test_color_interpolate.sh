for OBJECT in    'ball' 'bowl' 'plate' 'teddybear' 'sofa' 'mug' 'horse'  ### 'chair'

do

CUDA_VISIBLE_DEVICES=3 python test_e4c_object.py \
    --object "${OBJECT}" \
    --exp "colorpeel_e4c_maroon2olive_10000steps_multimap" \
    --inf_steps 25 \

done
