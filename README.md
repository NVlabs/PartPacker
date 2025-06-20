# PartPacker

![teaser](assets/teaser.gif)

### [Project Page](https://research.nvidia.com/labs/dir/partpacker/) | [Arxiv](https://arxiv.org/abs/2506.09980) | [Models](https://huggingface.co/nvidia/PartPacker) | [Demo](https://huggingface.co/spaces/nvidia/PartPacker)


This is the official implementation of *PartPacker: Efficient Part-level 3D Object Generation via Dual Volume Packing*.

Our model performs part-level 3D object generation from single-view images.

### Install

We rely on `torch` with CUDA installed correctly (tested with torch 2.5.1 + CUDA 12.1).

```bash
pip install -r requirements.txt

# if you prefer fixed version of dependencies:
pip install -r requirements.lock.txt

# by default we use torch's built-in attention, if you want to explicitly use flash-attn:
pip install flash-attn --no-build-isolation

# if you want to run data processing and vae inference, please install meshiki:
pip install meshiki
```

### Pretrained models

Download the pretrained models from huggingface, and put them in the `pretrained` folder.

```bash
mkdir pretrained
cd pretrained
wget https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt
wget https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt
```

### Inference

For inference, it takes ~10GB GPU memory (assuming float16).

```bash
# vae reconstruction of meshes
PYTHONPATH=. python vae/scripts/infer.py --ckpt_path pretrained/vae.pt --input assets/meshes/ --output_dir output/

# flow 3D generation from images
PYTHONPATH=. python flow/scripts/infer.py --ckpt_path pretrained/flow.pt --input assets/images/ --output_dir output/

# open local gradio app (single GPU)
python app.py

# open local gradio app with multi-GPU support
python app.py --multi
```

### Multi-GPU Support

The application supports multi-GPU inference for those who are lack of GPU memory.

- **Single GPU mode** (default): `python app.py`
- **Multi-GPU mode**: `python app.py --multi`

In multi-GPU mode:
- The flow model is placed on GPU 0
- The VAE model is placed on GPU 1 (if available)
- Automatic memory management and data transfer between GPUs
- Reduced memory pressure per GPU
- Better performance with 2 or more GPUs

If only one GPU is available, the system automatically falls back to single-GPU behavior even in multi-GPU mode.

### Data Processing

We provide a *Dual Volume Packing* implementation to process raw glb meshes into two separate meshes as proposed in the paper.

```bash
cd data
python bipartite_contraction.py ./example_mesh.glb
# the two separate meshes will be saved in ./output
```

### Acknowledgements

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

* [Dora](https://github.com/Seed3D/Dora)
* [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)
* [Trellis](https://github.com/microsoft/TRELLIS)

## Citation

```
@article{tang2024partpacker,
  title={Efficient Part-level 3D Object Generation via Dual Volume Packing},
  author={Tang, Jiaxiang and Lu, Ruijie and Li, Zhaoshuo and Hao, Zekun and Li, Xuan and Wei, Fangyin and Song, Shuran and Zeng, Gang and Liu, Ming-Yu and Lin, Tsung-Yi},
  journal={arXiv preprint arXiv:2506.09980},
  year={2025}
}
```
