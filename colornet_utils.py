import math
import numpy as np
import torch
import torch.nn as nn
from PIL import  ImageDraw, Image
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

shape_id_dict = {'circle':0, 'square':1, 'triangle':2, 'hexagon':3}
shape_id_dict_reverse = {v: k for k, v in shape_id_dict.items()}

    
color_id_dict = {'red':0, 'green':1, 'blue':2, 'yellow':3}
color_id_dict_reverse = {0:(255,0,0), 1:(0,255,0), 2:(0,0,255), 3:(255,255,0)}
# color_id_dict_reverse = {v: k for k, v in color_id_dict.items()}

def _compute_cosine(attention_maps: torch.Tensor,indices_to_alter: List[int],):
    x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
    cos_mask = torch.tril(torch.ones((len(indices_to_alter),len(indices_to_alter))),diagonal=-1).bool()
    cos_sim = F.cosine_similarity(x_attn[:,:,None], x_attn.t()[None,:,:])
    cos_dist = 1 - cos_sim[cos_mask].mean()

    return cos_dist

def _compute_cosine_seg(attention_maps: torch.Tensor,indices_to_alter: List[int], seg_maps=None):
    x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
    seg_maps_ = torch.cat(seg_maps).view(len(seg_maps),-1)
    return 1.0 - (F.cosine_similarity(x_attn, seg_maps_)).mean()

### TODO: hard code for now
def _compute_IoU_loss(attention_maps: torch.Tensor,indices_to_alter: List[int], seg_maps=None):
    x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()

    seg_maps_ = torch.cat((seg_maps[0],seg_maps[0])).view(2,-1)
    # seg_maps_ = torch.cat(seg_maps).view(len(seg_maps),-1)

    length=len(seg_maps_)
    loss_list=[(x_attn[i]*seg_maps_[i]).sum()/x_attn[i].sum() for i in range(length)]

    return 1 - sum(loss_list)/float(length)


# def _compute_cosine(attention_maps: torch.Tensor,indices_to_alter: List[int],):
#     x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
#     cos_mask = torch.tril(torch.ones((len(indices_to_alter),len(indices_to_alter))),diagonal=-1).bool()
#     cos_sim = F.cosine_similarity(x_attn[:,:,None], x_attn.t()[None,:,:])
#     cos_dist = cos_sim[cos_mask].mean()
#     return cos_dist

class E4TDataset(Dataset):
    def __init__(
            self,x0,x1, shape_size=256
    ):
        super().__init__()
        self.list=range(10000)
        self.start_color_embed=x0
        self.end_color_embed=x1
        self.shape_size=shape_size

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):

        start_color, end_color = torch.tensor([200,0,0], dtype=torch.float32), torch.tensor([255,0,0], dtype=torch.float32)

        color_lambda = torch.rand(1)
        color_fill = (start_color * (1-color_lambda) + end_color * color_lambda).to(torch.int)
        color_fill_embed = (self.start_color_embed * (1-color_lambda) + self.end_color_embed * color_lambda)

        shape_label = torch.randint(0, 4, (1,))
        shape = shape_id_dict_reverse[shape_label.item()]

        color_fill_tuple = tuple(color_fill.tolist())

        image_ = create_image_with_shapes(circle_diameter=self.shape_size, \
                                          fill_color=color_fill_tuple, shape=shape)
        image = np.array(image_.convert("RGB"))
        # image = self.processor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return dict(
            pixel_values=image,
            color_fill_embed=color_fill_embed,
            shape_label=shape_label,
        )



class E4TDataset_w_mask(Dataset):
    def __init__(
            self,x0,x1, shape_size=256
    ):
        super().__init__()
        self.list=range(10000)
        self.start_color_embed=x0
        self.end_color_embed=x1
        self.shape_size=shape_size

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        start_color, end_color = torch.tensor([200,0,0], dtype=torch.float32), torch.tensor([255,0,0], dtype=torch.float32)

        color_lambda = torch.rand(1)
        color_fill = (start_color * (1-color_lambda) + end_color * color_lambda).to(torch.int)
        color_fill_embed = (self.start_color_embed * (1-color_lambda) + self.end_color_embed * color_lambda)

        shape_label = torch.randint(0, 4, (1,))
        shape = shape_id_dict_reverse[shape_label.item()]

        color_fill_tuple = tuple(color_fill.tolist())

        image_, image_mask = create_image_with_shapes_fg_mask(circle_diameter=self.shape_size, \
                                          fill_color=color_fill_tuple, shape=shape)
        image = np.array(image_.convert("RGB"))

        seg_image = image_mask.resize((16,16))
        seg_img_data = np.asarray(seg_image).astype(bool)
        if len(seg_img_data.shape) >2:
            seg_img_data=seg_img_data[:,:,-1]
            seg_img_data = torch.from_numpy(seg_img_data).to(torch.float32).cuda().unsqueeze(0)
        # image = self.processor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return dict(
            pixel_values=image,
            mask=seg_img_data,
            color_fill_embed=color_fill_embed,
            shape_label=shape_label,
        )




class E4TDataset_RGB(Dataset):
    def __init__(
            self,x0,x1, shape_size=256
    ):
        super().__init__()
        self.list=range(10000)
        self.start_color_embed=x0
        self.end_color_embed=x1
        self.shape_size=shape_size

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        # start_color, end_color = torch.tensor([200,0,0], dtype=torch.float32), torch.tensor([255,0,0], dtype=torch.float32)

        # color_lambda = torch.rand(1)
        # color_fill = (start_color * (1-color_lambda) + end_color * color_lambda).to(torch.int)
        # color_fill_embed = (self.start_color_embed * (1-color_lambda) + self.end_color_embed * color_lambda)
        # color_fill_tuple = tuple(color_fill.tolist())

        color_label = torch.randint(0, 4, (1,))
        color_fill = color_id_dict_reverse[color_label.item()]

        shape_label = torch.randint(0, 4, (1,))
        shape = shape_id_dict_reverse[shape_label.item()]

        image_, image_mask = create_image_with_shapes_fg_mask(circle_diameter=self.shape_size, \
                                          fill_color=color_fill, shape=shape)
        image = np.array(image_.convert("RGB"))

        seg_image = image_mask.resize((16,16))
        seg_img_data = np.asarray(seg_image).astype(bool)
        if len(seg_img_data.shape) >2:
            seg_img_data=seg_img_data[:,:,-1]
            seg_img_data = torch.from_numpy(seg_img_data).to(torch.float32).cuda().unsqueeze(0)
        # image = self.processor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return dict(
            pixel_values=image,
            mask=seg_img_data,
            color_label=color_label,
            shape_label=shape_label,
        )



class ColorNet(ModelMixin, ConfigMixin):
    def __init__(self, hidden_size=384):
        super(ColorNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(768, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # First hidden layer
        self.fc3 = nn.Linear(hidden_size, 768)  # Second hidden layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function in the output layer
        return x


class ShapeNet(ModelMixin, ConfigMixin):
    def __init__(self, num_labels=4, embedding_dim=768, shape_init_embed=None):
        super(ShapeNet, self).__init__()
        self.embedding = nn.Embedding(num_labels, embedding_dim)
        self.embedding.weight= nn.Parameter(shape_init_embed[0])

    def forward(self, labels):
        return self.embedding(labels)
    
class ColorNet_RGB(ModelMixin, ConfigMixin):
    def __init__(self, num_labels=4, embedding_dim=768, color_init_embed=None):
        super(ColorNet_RGB, self).__init__()
        self.embedding = nn.Embedding(num_labels, embedding_dim)
        self.embedding.weight= nn.Parameter(color_init_embed[0])

    def forward(self, labels):
        return self.embedding(labels)
    


def create_image_with_shapes(circle_diameter = 256, fill_color = (150, 0, 0), shape='circle'):
    assert shape in ['circle', 'square', 'triangle', 'hexagon']

    # Create a new blank image with a white background
    width, height = 512, 512
    background_color = (255, 255, 255)
    image = Image.new("RGB", (width, height), background_color)
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Calculate the circle's diameter and position it in the center
    x_center = width // 2
    y_center = height // 2
    x_half = circle_diameter // 2  ### 128
    y_half = circle_diameter // 2

    # Draw the red circle
    if shape=='circle':
        draw.ellipse(
            [(x_center - x_half, y_center - y_half), (x_center + x_half, y_center + y_half)],
            fill=fill_color,
        )
    elif shape=='square':
        draw.rectangle(
            [(x_center - x_half, y_center - y_half), (x_center + x_half, y_center + y_half)],
            fill=fill_color,
        )
    elif shape=='triangle':
        top_point=(x_center, y_center - y_half*2.0/math.sqrt(3))
        left_point=(x_center-x_half, y_center + y_half/math.sqrt(3))
        right_point=(x_center+x_half, y_center + y_half/math.sqrt(3))

        draw.polygon([top_point, left_point, right_point], fill=fill_color)
    elif shape=='hexagon':
        points = [
            (x_center - x_half, y_center),
            (x_center - x_half*0.5, y_center - y_half*0.5*math.sqrt(3)),
            (x_center + x_half*0.5, y_center - y_half*0.5*math.sqrt(3)),
            (x_center + x_half, y_center ),
            (x_center + x_half*0.5, y_center + y_half*0.5*math.sqrt(3)),
            (x_center - x_half*0.5, y_center + y_half*0.5*math.sqrt(3)),
        ]

        draw.polygon(points, fill=fill_color)
    return image


def optim_init_colornet(color_encoder, x0, x1, y, step=500):
    optim=torch.optim.Adam(color_encoder.parameters())
    color_encoder, x0, x1, y = color_encoder.cuda(), x0.cuda(), x1.cuda(), y.cuda() 
    for loop_num in range(step):
        y_ = color_encoder(torch.cat([x0.unsqueeze(0),x1.unsqueeze(0)], dim=0))
        # print(y.shape)
        loss = ((y_ - y)**2).sum()
        if loop_num % 100 ==0:
            print(loss)

        loss.backward()
        optim.step()
        optim.zero_grad()

    return color_encoder


def create_image_with_shapes_fg_mask(circle_diameter = 256, fill_color = (150, 0, 0), shape='circle'):
    assert shape in ['circle', 'square', 'triangle', 'hexagon']

    # Create a new blank image with a white background
    width, height = 512, 512
    background_color = (255, 255, 255)
    image = Image.new("RGB", (width, height), background_color)
    image_mask = Image.new("RGB", (width, height), (0,0,0))
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(image_mask)

    # Calculate the circle's diameter and position it in the center
    x_center = width // 2
    y_center = height // 2
    x_half = circle_diameter // 2  ### 128
    y_half = circle_diameter // 2

    # Draw the red circle
    if shape=='circle':
        draw.ellipse(
            [(x_center - x_half, y_center - y_half), (x_center + x_half, y_center + y_half)],
            fill=fill_color,
        )
        draw_mask.ellipse(
            [(x_center - x_half, y_center - y_half), (x_center + x_half, y_center + y_half)],
            fill=(255,255,255),
        )
    elif shape=='square':
        draw.rectangle(
            [(x_center - x_half, y_center - y_half), (x_center + x_half, y_center + y_half)],
            fill=fill_color,
        )
        draw_mask.rectangle(
            [(x_center - x_half, y_center - y_half), (x_center + x_half, y_center + y_half)],
            fill=(255,255,255)
        )
    elif shape=='triangle':
        top_point=(x_center, y_center - y_half*2.0/math.sqrt(3))
        left_point=(x_center-x_half, y_center + y_half/math.sqrt(3))
        right_point=(x_center+x_half, y_center + y_half/math.sqrt(3))

        draw.polygon([top_point, left_point, right_point], fill=fill_color)
        draw_mask.polygon([top_point, left_point, right_point], fill=(255,255,255))

    elif shape=='hexagon':
        points = [
            (x_center - x_half, y_center),
            (x_center - x_half*0.5, y_center - y_half*0.5*math.sqrt(3)),
            (x_center + x_half*0.5, y_center - y_half*0.5*math.sqrt(3)),
            (x_center + x_half, y_center ),
            (x_center + x_half*0.5, y_center + y_half*0.5*math.sqrt(3)),
            (x_center - x_half*0.5, y_center + y_half*0.5*math.sqrt(3)),
        ]

        draw.polygon(points, fill=fill_color)
        draw_mask.polygon(points, fill=(255,255,255))

    return image, image_mask

