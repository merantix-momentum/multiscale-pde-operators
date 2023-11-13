<div align="center"> 

# Multiscale Neural Operators for Solving Time-Independent PDEs

</div>

**TL;DR:** We study how to solve time-independent Partial Differential Equations on large meshes and introduce a novel graph rewiring technique for this.

# 

> [**Multiscale Neural Operators for Solving Time-Independent PDEs**](https://arxiv.org/abs/2311.05964)
>
> by Winfried Ripken<sup>1</sup> \*, Lisa Coiffard<sup>1</sup> \*, Felix Pieper<sup>1</sup> \* and Sebastian Dziadzio<sup>2</sup>.
> 
> <sup>1</sup> [Merantix Momentum](https://en.merantix-momentum.com), <sup>2</sup> [Tübingen AI Center](https://tuebingen.ai).
> 
> (\*) equal contribution.
<br>

#

### Requirement

- Run ```pip install -e .``` and ```pip install -r requirements.txt``` in root folder before using.
- Install gcloud CLI and authenticate:
```
gcloud auth login
gcloud auth application-default login
```
- The BSMS operator needs intel mkl installed, best installed via conda.
- Check ```data/download_data.py``` for downloading data

We recommend installing our repository using 
```
pip install . -e
```
### Training

Start training run:
```
python -m multiscale_operator.model.trainer --config-name={config_name}
```


## Acknowledgement

We integrated 3 datasets:
- Darcy Flow in 2D
Data courtesy under [PDE Bench](https://github.com/pdebench/PDEBench).
- Magnetic Field for electric motor simulations
Data courtesy under [Multiphysics Optimization](https://arxiv.org/abs/2309.13179).
- 2D Poisson Equation for Magnetostatics:
Data courtesy under [GNN BVP Solver](https://github.com/merantix-momentum/gnn-bvp-solver).

Our operator implementations are based on the following public repositories:
- [Mesh Graph Nets in Pytorch](https://github.com/echowve/meshGraphNets_pytorch/tree/master)
- [Bi-stride Multi-Scale GNN](https://github.com/Eydcao/BSMS-GNN)

Please cite the relevant publications. For our datasets:

- Darcy Flow:
```
@inproceedings{PDEBench2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
title = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
year = {2022},
booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
url = {https://arxiv.org/abs/2210.07182}
}
```

- Motor Dataset:
```
@article{botache2023enhancing,
  title={Enhancing Multi-Objective Optimization through Machine Learning-Supported Multiphysics Simulation},
  author={Botache, Diego and Decke, Jens and Ripken, Winfried and Dornipati, Abhinay and G{\"o}tz-Hahn, Franz and Ayeb, Mohamed and Sick, Bernhard},
  journal={arXiv preprint arXiv:2309.13179},
  year={2023}
}
```

- Magnetostatics Dataset:
```
@inproceedings{lotzsch2022learning,
  title={Learning the Solution Operator of Boundary Value Problems using Graph Neural Networks},
  author={L{\"o}tzsch, Winfried and Ohler, Simon and Otterbach, Johannes},
  booktitle={ICML 2022 2nd AI for Science Workshop},
  year={2022}
}
```

For the benchmarked methods:
- BSMS:
```
@inproceedings{cao2022bi,
  title={Bi-stride multi-scale graph neural network for mesh-based physical simulation},
  author={Cao, Yadi and Chai, Menglei and Li, Minchen and Jiang, Chenfanfu},
  booktitle={International conference on machine learning},
  organization={PMLR},
  year={2023}
}
```

- Perceiver IO:
```
@inproceedings{jaegle2021perceiver,
  title={Perceiver IO: A General Architecture for Structured Inputs \& Outputs},
  author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

- Mesh Graph Nets (MGN):
```
@inproceedings{pfaff2020learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
