
This work is built on the repository https://github.com/Chan-Sun/IFSDD. Thanks for making it open source!

Check source code in the folder `defect`

## installation

1. Create conda environment

```shell
conda create --name defect python=3.8 -y
conda activate defect
```

2. Install `Pytorch` with cuda

```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install `MMCV` and `MMCLS`

```shell
pip install -U openmim
mim install mmcv-full==1.4.6
pip install mmcls==0.16
```

4. Build the source code of dowloaded packages and the current one

```shell
cd ./packages

cd ./mmdetection
pip install -r requirements.txt
pip install -e .
```

5. (Optional) install tensorboard for monitoring training and testing

```shell
pip install tensorboard==1.15
```

To use Jupiter Notebook install the following package:
```shell
conda install -n defect ipykernel --update-deps --force-reinstall -y
```

In case weights of ResNet are not found, copy the file ```weights/resnet101.pth``` to the hidden directory ```~/.torch/models```

