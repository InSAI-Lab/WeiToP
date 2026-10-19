# WeiToP & WeiAD

[![arXiv](https://img.shields.io/badge/arXiv-2605.20551-b31b1b.svg)](https://arxiv.org/abs/2605.20551)

**[Faster or Stronger: Towards Flexible Visual Place Recognition via Weighted Aggregation and Token Pruning](https://arxiv.org/abs/2605.20551)**

**WeiAD** (Weighted Aggregated Descriptor) produces global place descriptors through importance-weighted optimal transport aggregation. **WeiToP** distils aggregation-induced token importance into a lightweight predictor, enabling early token pruning during inference. Both components are trained jointly, allowing the token retention ratio to be adjusted at inference time to balance recognition accuracy and computational cost without retraining.

## 📋 TODO

- [x] Release WeiAD and WeiToP models.
- [x] Release training code and the GSV-Cities dataloader.
- [ ] Release evaluation code and benchmark dataloaders.
- [ ] Add detailed training and evaluation instructions.
- [ ] Release pretrained checkpoints.

## 📁 Dataset

We train on [GSV-Cities](https://github.com/amaralibey/gsv-cities). Follow the dataset repository's download instructions and extract the images and metadata into the following structure:

```text
gsv-cities/
├── Dataframes/
│   ├── Bangkok.csv
│   ├── Boston.csv
│   └── ...
└── Images/
    ├── Bangkok/
    │   └── *.jpg
    ├── Boston/
    │   └── *.jpg
    └── ...
```

Keep the city names and image filenames unchanged. Pass the directory containing `Dataframes/` and `Images/` to `--data-root`. The dataloader samples four images per place from the 23 training cities.

## 🚀 Train

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start training on GSV-Cities:

```bash
python train.py \
    --data-root /path/to/gsv-cities \
    --output-dir runs/weitop \
    --device cuda:0
```

DINOv2 pretrained weights are loaded through PyTorch Hub. Training runs for four epochs by default, with 60 places and four images per place in each batch. Adjust `--batch-size` and `--num-workers` for your hardware. Training checkpoints and logs are saved under `--output-dir`.

## 🙏 Acknowledgements

Our implementation builds on [SALAD](https://github.com/serizba/salad), [DINOv2](https://github.com/facebookresearch/dinov2), [GSV-Cities](https://github.com/amaralibey/gsv-cities) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork). We thank their authors for making their code and datasets available. We also use [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning).

## 📖 Cite

If you find this work useful, please cite:

```bibtex
@misc{zeng2026weitop,
  title={Faster or Stronger: Towards Flexible Visual Place Recognition via Weighted Aggregation and Token Pruning},
  author={Zichao Zeng and June Moh Goo and Junwei Zheng and Weijia Fan and Jiaming Zhang and Rainer Stiefelhagen and Jan Boehm},
  year={2026},
  eprint={2605.20551},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.20551}
}
```
