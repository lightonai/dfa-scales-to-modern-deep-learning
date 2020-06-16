# Scaling-up DFA -- Neural Rendering

## Implementation

Our implementation is based on: https://github.com/krrish94/nerf-pytorch

## Organization

`train_nerf.py` contains all the training logic to train a NeRF model. `test_nerf.py` can be used on a trained model to generate the PSNR scores from Table A.5, then averaged for Table 1. `render_nerf.py` is used to generate the renders of the paper and the video. `tiny_nerf.py` is used for Figure A.1. Other python files are utilities to power NeRF. Note that our DFA models are defined in `nerf/models.py`. 

`config` contains the configuration files used for the paper, and `pretrained-models` our trained models. 

## Reproducibility 

Data should be first obtained from the official paper website: http://www.matthewtancik.com/nerf

To train a NeRF on the Fern scene of LLFF-Synthetic:
```bash
python train_nerf.py --config config/real-fern.xml
```

It can then be evaluated with:
```bash
python test_nerf.py --config config/real-fern.xml --checkpoint checkpoint.ckpt
```

Render images can be generated with:

```bash
python render_nerf.py --config config/real-fern.xml --checkpoint checkpoint.ckpt
```

They can then be assembled into a `.gif` using `ImageMagick` for instance. 

Documentation is available within each script for additionnal parameters (such as multi-GPU strategies, or NeRF-Dual).
