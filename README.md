<div align="center">
<h1>[ECCV24] Mask2Map <img src="assets/map.png" width="30"></h1>
<h3>Vectorized HD Map Construction Using Bird’s Eye View Segmentation Masks</h3>

Sewhan Choi<sup>1</sup> \*, Jungho Kim<sup>1</sup> \*, Hongjae Shin<sup>1</sup>, Junwon Choi<sup>2</sup> \**
 
<sup>1</sup> Hanyang University, Korea <sup>2</sup> Seoul National University, Korea

(\*) equal contribution, (<sup>**</sup>) corresponding author.

ArXiv Preprint ([arXiv 2208.05736](https://arxiv.org/abs/2308.05736))
<!-- [ECCV'24](??) -->

</div>


## Introduction

![overall](assets/overall.png "overall")

In this paper, we introduce Mask2Map, a novel end-to-end online HD map construction method designed for autonomous driving applications. Our approach focuses on predicting the class and ordered point set of map instances within a scene, represented in the bird's eye view (BEV).
Mask2Map consists of two primary components: the Instance-Level Mask Prediction Network (IMPNet) and the Mask-Driven Map Prediction Network (MMPNet). IMPNet generates Mask-Aware Queries and BEV Segmentation Masks to capture comprehensive semantic information globally. Subsequently, MMPNet enhances these query features using local contextual information through two submodules: the Positional Query Generator (PQG) and the Geometric Feature Extractor (GFE). PQG extracts instance-level positional queries by embedding BEV positional information into Mask-Aware Queries, while GFE utilizes BEV Segmentation Masks to generate point-level geometric features.
However, we observed limited performance in Mask2Map due to inter-network inconsistency stemming from different predictions to Ground Truth (GT) matching between IMPNet and MMPNet. To tackle this challenge, we propose the Inter-network Denoising Training method, which guides the model to denoise the output affected by both noisy GT queries and perturbed BEV Segmentation Masks.


## Demo

![demo](assets/demo.gif "demo")

## Acknowledgements

Mask2Map is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [MapTR](https://github.com/hustvl/MapTR), [BEVFusion](https://github.com/mit-han-lab/bevfusion), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet), [GKT](https://github.com/hustvl/GKT), [VectorMapNet](https://github.com/Mrmoore98/VectorMapNet_code).

## Citation
If you find Mask2Map is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{Mask2Map,
  title={Mask2Map: Vectorized HD Map Construction Using Bird’s Eye View Segmentation Masks},
  author={Choi, Sewhan and Kim, Jungho and Shin, Hongjae and Choi, Jun Won},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
