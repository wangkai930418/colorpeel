import torch
from diffusers import DiffusionPipeline 
### NOTE: consider to change into stable diffusion pipeline to avoid safety checker
from pipelines.stable_diffusion_pipeline import StableDiffusionPipeline 
from colornet_utils import create_image_with_shapes, _compute_cosine, ColorNet, optim_init_colornet

import os
from datetime import datetime
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--exp", type=str, default='colorpeel_e4c')
	parser.add_argument("--inf_steps", type=int, default=100)
	parser.add_argument("--seeds", type=int, default=42)
	parser.add_argument("--scale", type=float, default=6.0)
	parser.add_argument("--samples", type=int, default=3)
	parser.add_argument('--object', type=str, default='chair')

	args = parser.parse_args()
	
	prompts = [
		"a photo of {object_placeholder} in <c*> color",
		# "a photo of <s2*> shape in <c*> color",
		# "a photo of <s3*> shape in <c*> color",
		# "a photo of <s4*> shape in <c*> color",
		# "a photo of <s1*> shape in <c1*> color",
		# "a photo of <s1*> shape in <c2*> color",
		# "a photo of <s1*> shape in <c3*> color",
		# "a photo of <s1*> shape in <c4*> color",
		# "a photo of <s2*> shape in <c1*> color",
		# "a photo of <s2*> shape in <c2*> color",
		# "a photo of <s2*> shape in <c3*> color",
		# "a photo of <s2*> shape in <c4*> color",
		]

	model_path = f'models/{args.exp}/'
	LOG_DIR=f''

	model_id = f"{model_path+LOG_DIR}"

	out_path = f'output/{args.exp}_{args.inf_steps}_{args.seeds}_{args.scale}_{args.samples}_personlized/'

	isExist = os.path.exists(out_path)
	if not isExist:
  		os.makedirs(out_path)
		
	osExist = os.path.exists(f'{out_path}/{LOG_DIR}')

	if not osExist:
  		os.makedirs(f'{out_path}/{LOG_DIR}')
	
	# pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
	pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
	### NOTE: disable the safety checker
	pipe.safety_checker = None

	pipe.unet.load_attn_procs(f"{model_id}", weight_name="pytorch_custom_diffusion_weights.bin")

	pipe.load_textual_inversion(f"{model_id}", weight_name="<s1*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s2*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s3*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s4*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c*>.bin")
	
	### NOTE: new code to import the text encoder
	text_encoder = pipe.text_encoder
	tokenizer = pipe.tokenizer

	token_embeds = text_encoder.get_input_embeddings().weight.data
	color_x0 = token_embeds[49400]
	color_x1 = token_embeds[49401]
	color_encoder = ColorNet(hidden_size=1568)
	color_enc_path=f"{model_path}/color_encoder.pth"
	color_encoder.load_state_dict(torch.load(color_enc_path))
	color_encoder = color_encoder.cuda()
	# pipe.load_textual_inversion(f"{model_id}", weight_name="<c1*>.bin")
	# pipe.load_textual_inversion(f"{model_id}", weight_name="<c2*>.bin")
	# pipe.load_textual_inversion(f"{model_id}", weight_name="<c3*>.bin")
	# pipe.load_textual_inversion(f"{model_id}", weight_name="<c4*>.bin")
	color_token='<c*>'
	color_token_id = tokenizer.convert_tokens_to_ids(color_token)
	### NOTE: put the new embedding over here.
	# token_embeds[color_token_id]=torch.zeros(768)

	for prompt_ in prompts:
		prompt = prompt_.format(object_placeholder=args.object)
		# n_samples = args.samples
		path = f'{out_path}/{LOG_DIR}/{prompt}_{datetime.now()}'
		os.mkdir(path)
        
		# prompt_ids = tokenizer(
        #     prompt,
        #     truncation=True,
        #     padding="max_length",
        #     max_length=tokenizer.model_max_length,
        #     return_tensors="pt",
        # ).input_ids
        
		# inputs_embeds = text_encoder.get_input_embeddings()(prompt_ids.to("cuda"))
		# for ind in range(n_samples):
		# for color_lambda in torch.arange(0, 1.1, 0.1):
		# for color_lambda in torch.arange(0.51, 0.61, 0.01):

		# for color_lambda in torch.arange(0.51, 0.52, 0.001):
		for color_lambda in torch.arange(0.0, 1.00, 0.01):
			pre_color_embed = (color_x0 * (1-color_lambda) + color_x1 * color_lambda)
			post_color_embed=color_encoder(pre_color_embed)
			token_embeds[color_token_id]=post_color_embed
			
			gen = torch.Generator(device="cuda").manual_seed(args.seeds)

			out = pipe(prompt, num_inference_steps=args.inf_steps, generator=gen, guidance_scale=args.scale,eta=1.0,)

			out.images[0].save(f"{path}/{prompt}_{int(color_lambda*1000):04}.png")
			# out.images[0].save(f"{path}/{prompt}_{int(color_lambda*100)}.png")

if __name__ == "__main__":
    main()