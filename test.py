import torch
from diffusers import DiffusionPipeline 
### NOTE: consider to change into stable diffusion pipeline to avoid safety checker
from diffusers import StableDiffusionPipeline 

import os
from datetime import datetime
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--exp", type=str, default='colorpeel_exp11_3d_2s4c_w2c_cos')
	parser.add_argument("--inf_steps", type=int, default=100)
	parser.add_argument("--seeds", type=int, default=42)
	parser.add_argument("--scale", type=float, default=6.0)
	parser.add_argument("--samples", type=int, default=3)

	args = parser.parse_args()
	
	prompts = [
		"a photo of <s1*> shape in <c1*> color",
		"a photo of <s1*> shape in <c2*> color",
		"a photo of <s1*> shape in <c3*> color",
		"a photo of <s1*> shape in <c4*> color",
		"a photo of <s2*> shape in <c1*> color",
		"a photo of <s2*> shape in <c2*> color",
		"a photo of <s2*> shape in <c3*> color",
		"a photo of <s2*> shape in <c4*> color",
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
	
	pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
	### NOTE: disable the safety checker
	pipe.safety_checker = None

	pipe.unet.load_attn_procs(f"{model_id}", weight_name="pytorch_custom_diffusion_weights.bin")

	pipe.load_textual_inversion(f"{model_id}", weight_name="<s1*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s2*>.bin")
	
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c1*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c2*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c3*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c4*>.bin")

	for prompt in prompts:

		gen = torch.Generator(device="cuda").manual_seed(args.seeds)
		n_samples = args.samples
		path = f'{out_path}/{LOG_DIR}/{prompt}_{datetime.now()}'
		os.mkdir(path)

		for ind in range(n_samples):
			out = pipe(prompt, num_inference_steps=args.inf_steps, generator=gen, guidance_scale=args.scale,eta=1.0,)
			out.images[0].save(f"{path}/{prompt}_{ind}.png")

if __name__ == "__main__":
    main()