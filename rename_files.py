import os
import shutil

org_path='output/colorpeel_e4c_maroon2olive_25_42_6.0_3_personlized/'+\
            'a photo of chair in <c*> color_2024-02-04 14:24:50.604366/'
            # 'a photo of chair in <c*> color_2024-02-04 13:45:22.991242/'

tag_pth='output/colorpeel_e4c_maroon2olive/chair/'

os.makedirs(tag_pth, exist_ok=True)

image_names = sorted(os.listdir(org_path))
for ind in range(len(image_names)):
    id = image_names[ind].split('.')[0].split('_')[-1]
    if len(id) == 3:
        new_id = str(int(id)*10) +'.png'
    else:
        new_id = str(int(id)*100)+ '.png'
        # print(int(id)*100)

    if len(id) == 1:
        new_id = '0' + new_id
    # new_id = str(int(id)*100)+ '.png'

    print(new_id)
    source_file=os.path.join(org_path, image_names[ind])
    destination_file=os.path.join(tag_pth, new_id)
    shutil.copy(source_file, destination_file)