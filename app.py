import os
import argparse
from datetime import datetime

import cv2
import gradio as gr
import kiui
import numpy as np
import rembg
import torch
import torch.nn as nn
import trimesh

# Memory optimization setting
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    # running on Hugging Face Spaces
    import spaces

except ImportError:
    # running locally, use a dummy space
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration

            def __call__(self, func):
                return func


# download checkpoints
from huggingface_hub import hf_hub_download

from flow.configs.schema import ModelConfig
from flow.model import Model
from flow.utils import get_random_color, recenter_foreground
from vae.utils import postprocess_mesh

flow_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="flow.pt")
vae_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="vae.pt")

TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
MAX_SEED = np.iinfo(np.int32).max
bg_remover = rembg.new_session()

# model config
model_config = ModelConfig(
    vae_conf="vae.configs.part_woenc",
    vae_ckpt_path=vae_ckpt_path,
    qknorm=True,
    qknorm_type="RMSNorm",
    use_pos_embed=False,
    dino_model="dinov2_vitg14",
    hidden_dim=1536,
    flow_shift=3.0,
    logitnorm_mean=1.0,
    logitnorm_std=1.0,
    latent_size=4096,
    use_parts=True,
)

# Multi-GPU setup
def setup_multi_gpu():
    """Configures multiple GPUs and assigns devices."""
    if not torch.cuda.is_available():
        return {'primary': 'cpu', 'secondary': 'cpu', 'num_gpus': 0}
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus >= 2:
        # Separate flow and VAE models if 2 or more GPUs are available
        primary_device = 'cuda:0'  # For flow model
        secondary_device = 'cuda:1'  # For VAE model
        print(f"Enabling model parallelism: Flow -> {primary_device}, VAE -> {secondary_device}")
    else:
        # Single GPU case
        primary_device = 'cuda:0'
        secondary_device = 'cuda:0'
        print(f"Using single GPU: {primary_device}")
    
    return {
        'primary': primary_device,
        'secondary': secondary_device, 
        'num_gpus': num_gpus
    }

class MultiGPUModel(nn.Module):
    """Model wrapper for multi-GPU support."""
    def __init__(self, model_config, gpu_config):
        super().__init__()
        self.gpu_config = gpu_config
        self.config = model_config
        
        # Create the base model
        self.base_model = Model(model_config).eval()
        
        # Place flow model on the primary GPU
        if hasattr(self.base_model, 'flow'):
            self.base_model.flow = self.base_model.flow.to(gpu_config['primary']).bfloat16()
        
        # Place VAE model on the secondary GPU (if available)
        if hasattr(self.base_model, 'vae'):
            self.base_model.vae = self.base_model.vae.to(gpu_config['secondary']).bfloat16()
        
        # Place other components on the primary GPU
        for name, module in self.base_model.named_children():
            if name not in ['flow', 'vae']:
                module.to(gpu_config['primary']).bfloat16()
    
    def forward(self, data, num_steps=50, cfg_scale=7):
        """Performs inference, managing data transfer between devices."""
        # Move input data to the appropriate device
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(self.gpu_config['primary'])
        
        # Clear memory
        if self.gpu_config['num_gpus'] > 0:
            torch.cuda.empty_cache()
        
        # Generate latent representation with the flow model
        with torch.inference_mode():
            results = self.base_model(data, num_steps=num_steps, cfg_scale=cfg_scale)
        
        return results
    
    def vae_decode(self, data, resolution=384):
        """Performs VAE decoding, transferring data between devices as needed."""
        # Move data to the VAE's device
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(self.gpu_config['secondary'])
        
        # Clear memory
        if self.gpu_config['num_gpus'] > 0:
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            results = self.base_model.vae(data, resolution=resolution)
        
        return results

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--multi', action='store_true', help='Enable multi-GPU support')
args = parser.parse_args()

# Initialize GPU configuration and model based on arguments
if args.multi:
    gpu_config = setup_multi_gpu()
    model = MultiGPUModel(model_config, gpu_config)
    multi_gpu_enabled = True
else:
    gpu_config = {'num_gpus': 1 if torch.cuda.is_available() else 0}
    model = Model(model_config).eval().cuda().bfloat16()
    multi_gpu_enabled = False

# load weight
ckpt_dict = torch.load(flow_ckpt_path, weights_only=True)
if multi_gpu_enabled:
    model.base_model.load_state_dict(ckpt_dict, strict=True)
else:
    model.load_state_dict(ckpt_dict, strict=True)


# get random seed
def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed


# process image
@spaces.GPU(duration=10)
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bg removal if there is no alpha channel
        image = rembg.remove(image, session=bg_remover)  # [H, W, 4]
    mask = image[..., -1] > 0
    image = recenter_foreground(image, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_AREA)
    return image


# process generation
@spaces.GPU(duration=90)
def process_3d(
    input_image, num_steps=50, cfg_scale=7, grid_res=384, seed=42, simplify_mesh=False, target_num_faces=100000
):

    # seed
    kiui.seed_everything(seed)
    
    # Display GPU memory usage for multi-GPU mode
    if multi_gpu_enabled and gpu_config['num_gpus'] > 0:
        for i in range(gpu_config['num_gpus']):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated Memory {memory_allocated:.2f}GB, Reserved {memory_reserved:.2f}GB")

    # output path
    os.makedirs("output", exist_ok=True)
    output_glb_path = f"output/partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"

    # input image (assume processed to RGBA uint8)
    image = input_image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float()
    
    if not multi_gpu_enabled:
        image_tensor = image_tensor.cuda()

    data = {"cond_images": image_tensor}

    if multi_gpu_enabled:
        # Multi-GPU processing
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)
        latent = results["latent"]

        # Query mesh - process each part separately to save memory
        data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
        data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

        # Generate mesh for part 0
        results_part0 = model.vae_decode(data_part0, resolution=grid_res)
        
        # Clear memory
        if gpu_config['num_gpus'] > 0:
            torch.cuda.empty_cache()
        
        # Generate mesh for part 1
        results_part1 = model.vae_decode(data_part1, resolution=grid_res)
        
        # Clear memory
        if gpu_config['num_gpus'] > 0:
            torch.cuda.empty_cache()
    else:
        # Single GPU processing (original code)
        with torch.inference_mode():
            results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)

        latent = results["latent"]

        # query mesh
        data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
        data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

        with torch.inference_mode():
            results_part0 = model.vae(data_part0, resolution=grid_res)
            results_part1 = model.vae(data_part1, resolution=grid_res)

    if not simplify_mesh:
        target_num_faces = -1

    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
    parts = mesh_part0.split(only_watertight=False)

    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
    parts.extend(mesh_part1.split(only_watertight=False))

    # some parts only have 1 face, seems a problem of trimesh.split.
    parts = [part for part in parts if len(part.faces) > 10]

    # split connected components and assign different colors
    for j, part in enumerate(parts):
        # each component uses a random color
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    # export the whole mesh
    mesh.export(output_glb_path)

    return output_glb_path


# gradio UI
if multi_gpu_enabled:
    _TITLE = """PartPacker: Efficient Part-level 3D Object Generation via Dual Volume Packing (Multi-GPU Supported)"""
    _DESCRIPTION = f"""
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/dir/partpacker/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/NVlabs/PartPacker"><img src='https://img.shields.io/github/stars/NVlabs/PartPacker?style=social'/></a>
</div>

* GPU configuration: {gpu_config['num_gpus']} GPUs detected
* Each part is visualized with a random color and can be separated in the GLB file
* If the output is not satisfactory, try a different random seed!
* If you run out of memory, reduce the Grid resolution
"""
else:
    _TITLE = """PartPacker: Efficient Part-level 3D Object Generation via Dual Volume Packing"""
    _DESCRIPTION = """
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/dir/partpacker/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/NVlabs/PartPacker"><img src='https://img.shields.io/github/stars/NVlabs/PartPacker?style=social'/></a>
</div>

* Each part is visualized with a random color, and can be separated in the GLB file.
* If the output is not satisfactory, please try different random seeds!
"""

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# " + _TITLE)
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                # input image
                input_image = gr.Image(label="Input Image", type="filepath")  # use file_path and load manually
                seg_image = gr.Image(label="Segmentation Result", type="numpy", interactive=False, image_mode="RGBA")
            with gr.Accordion("Settings", open=True):
                # inference steps
                num_steps = gr.Slider(label="Inference steps", minimum=1, maximum=100, step=1, value=50)
                # cfg scale
                cfg_scale = gr.Slider(label="CFG scale", minimum=2, maximum=10, step=0.1, value=7.0)
                # grid resolution - adjust default based on multi-GPU mode
                default_grid_res = 256 if multi_gpu_enabled else 384
                min_grid_res = 192 if multi_gpu_enabled else 256
                input_grid_res = gr.Slider(label="Grid resolution", minimum=min_grid_res, maximum=512, step=1, value=default_grid_res)
                # random seed
                with gr.Row():
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                # simplify mesh
                with gr.Row():
                    simplify_mesh = gr.Checkbox(label="Simplify mesh", value=False)
                    target_num_faces = gr.Slider(
                        label="Face number", minimum=10000, maximum=1000000, step=1000, value=100000
                    )
                # gen button
                button_gen = gr.Button("Generate")

        with gr.Column(scale=1):
            # glb file
            output_model = gr.Model3D(label="Geometry", height=512)

    with gr.Row():
        gr.Examples(
            examples=[
                ["assets/images/rabbit.png"],
                ["assets/images/robot.png"],
                ["assets/images/teapot.png"],
                ["assets/images/barrel.png"],
                ["assets/images/cactus.png"],
                ["assets/images/cyan_car.png"],
                ["assets/images/pickup.png"],
                ["assets/images/swivelchair.png"],
                ["assets/images/warhammer.png"],
            ],
            fn=process_image,  # still need to click button_gen to get the 3d
            inputs=[input_image],
            outputs=[seg_image],
            cache_examples=False,
        )

    button_gen.click(process_image, inputs=[input_image], outputs=[seg_image]).then(
        get_random_seed, inputs=[randomize_seed, seed], outputs=[seed]
    ).then(
        process_3d,
        inputs=[seg_image, num_steps, cfg_scale, input_grid_res, seed, simplify_mesh, target_num_faces],
        outputs=[output_model],
    )

block.launch()
