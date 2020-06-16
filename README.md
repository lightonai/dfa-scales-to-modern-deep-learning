# <img src="_static/lighton_small.png" width=60/> Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Twitter](https://img.shields.io/twitter/follow/LightOnIO?style=social)](https://twitter.com/LightOnIO)

Code for our paper [Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures](https://arxiv.org/). 

We study the applicability of Direct Feedback Alignment (DFA) to neural view synthesis, recommender systems, geometric learning, and natural language processing. At variance with common beliefs, we show that challenging tasks can be tackled in the absence of weight transport.

## Requirements

- A `requirements.txt` file is available at the root of this repository, specifying the required packages for all of our experiments;
- Our DFA implementation, `TinyDFA`, is pip-installable: from the `TinyDFA` folder, run `pip install .`; 
- `tsnecuda` may require installation from source: see the [tsne-cuda repository](https://github.com/CannyLab/tsne-cuda) for details;
- Neural rendering datasets can be found on the [NeRF website](http://www.matthewtancik.com/nerf)--other datasets will be automatically fetched.

## Citation

If you found this code useful in your research, please consider citing:

```bibtex
@article{launay2020dfascaling,
  title={Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures},
  author={Launay, Julien and Iacopo, Poli and Francois, Boniface and Krzakala, Florent},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2020}
}
```

## Reproducing our results

- Instructions for reproduction are given within each task folder, in the associated `README.md` file. 

## <img src="_static/lighton_cloud_small.png" width=120/> About LightOn/LightOn Cloud

LightOn develops a light-based technology required in large scale artificial intelligence computations. Our ambition is to significantly reduce the time and energy required to make sense of the world around us.

Please visit https://cloud.lighton.ai/ for more information on accessing our technology.
