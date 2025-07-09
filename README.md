# SVFormer: A Direct Training Spiking Transformer for Efficient Video Action Recognition
This repo is an SNN extension of [UniFormer](https://github.com/Sense-X/UniFormer/tree/main/video_classification), based on which we explored directly-trained spiking neural networks for video action recognition.

## Usage

### Installation

First of all, you need to install slowfast. Please follow the installation instructions in [INSTALL.md](INSTALL.md). 

Then, you need to install `spikingjelly==0.0.0.0.14` for SNN development and training.

Besides, you may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.

### Pretrained models

We release the checkpoints on Baidu Cloud: [ckpt (bv4p)](https://pan.baidu.com/s/1Inrw3lXmGShLMhgy0YeEnA)

Download them and place them in the corresponding [checkpoints](./exp/svformer_ucf101_scratch/checkpoints/) folder.

### Training

Simply run the training scripts in [exp](exp) as followed:

   ```shell
   bash ./exp/svformer_ucf101_scratch/run.sh
   ```

**[Note]:**

- During training, we follow the [SlowFast](https://github.com/facebookresearch/SlowFast) repository and randomly crop videos for validation. For accurate testing, please follow our testing scripts.

- For more config details, you can read the comments in `slowfast/config/defaults.py`.

- Existed folders in [exp](exp) contain an example. You can make a new directory for experiment.

### Testing

We provide testing example as followed:

```shell
bash ./exp/svformer_ucf101_scratch/test.sh
```

Specifically, we need to create our new config for testing and run multi-crop/multi-clip test:

1. Copy the training config file `config.yaml` and create new testing config `test.yaml`.

2. Change the hyperparameters of data (in `test.yaml` or `test.sh`):

   ```yaml
   DATA:
     TRAIN_JITTER_SCALES: [224, 224]
     TEST_CROP_SIZE: 224
   ```

3. Set the number of crops and clips (in `test.yaml` or `test.sh`):

   Multi-clip testing (the numbers can be modified)

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 5
   TEST.NUM_SPATIAL_CROPS 3
   ```

4. You can also set the checkpoint path via:

   ```shell
   TEST.CHECKPOINT_FILE_PATH your_model_path
   ```

### Define and train your own SNN model

1. Define your SNN model like `./slowfast/models/uniformer2d_psnn_try.py`, and others in the same folder.
2. Make some modifications in `./slowfast/models/__init__.py`, `./slowfast/models/build.py`, `./slowfast/config/defaults.py`, `./tools/train_net.py`, etc. (see examples in the corresponding files for existed snn model)
3. Train and test the model by following the above instructions.

**[Note]:**

- In this repo, `uniformer2d_psnn` is actually `svformer`.

##  Cite Uniformer

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@inproceedings{yu2024svformer,
  title={Svformer: a direct training spiking transformer for efficient video action recognition},
  author={Yu, Liutao and Huang, Liwei and Zhou, Chenlin and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  booktitle={International Workshop on Human Brain and Artificial Intelligence},
  pages={161--180},
  year={2024},
  organization={Springer}
}
```

## Acknowledgement

This repository is developed based on several repositories: [UniFormer](https://github.com/Sense-X/UniFormer/tree/main/video_classification), [SlowFast](https://github.com/facebookresearch/SlowFast), [SpikingJelly ](https://github.com/fangwei123456/spikingjelly), [syops-counter](https://github.com/iCGY96/syops-counter), and others.
Thanks for their efforts.
