## Examples with [Maté 🧉](https://github.com/ilex-paraguariensis/yerbamate/tree/lightning) [PyTorch Lightning⚡](https://github.com/Lightning-AI/lightning) [huggingface 🤗](https://github.com/huggingface/transformers) 
A vision respository of deep learning tasks with ViT and CNN models. Maté 🧉 supports pretrianed models from any installed python package, including [huggingface 🤗](https://github.com/huggingface/transformers)  and [torchvision](https://github.com/pytorch/vision) models. All the ViT models are sourced from [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch) repository. You can use Maté 🧉 for classification, image generation or loading pretrained models. This project requires pytorch, pytorch-lightning, huggingface-torch, torch-vision. Maté 🧉 is a simple wrapper around pytorch-lightning, so you can use any pytorch-lightning trainer arguments. Maté 🧉 also supports distributed training, so you can train your models on multiple GPUs or datasets. Maté 🧉 is still in development, so please report any bugs or feature requests.

<!--
## Getting started

You can run mate on your local machine or run a jupyter notebook on google colab. 

## Colab

You can run the notebook on colab by clicking on the following badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lrhm/cifar-classification/blob/main/vit_mate.ipynb)

A Jupiter notebook is also available in the repository.

-->
## Install locally
First install mate by:
```bash
  git clone https://github.com/ilex-paraguariensis/yerbamate -b v2
```
Then move to the yerbamate directory
```
cd yerbamate
python install.py
```
Then install the requirements:
```bash
pip install -r requirements.txt
```
Then clone this repo:
```bash
 git clone https://github.com/ilex-paraguariensis/vision -b v2
```
And test that everything is working by:
```bash	
mate list models
```
Should output:
```bash
	vit
	lightning
	jax
	keras
```

<!--
First, install the dev version of Maté from lightning branch [link](https://github.com/ilex-paraguariensis/yerbamate/tree/lightning/tree/lightning).

Then, install the dependencies:
```bash
pip install -r project/requirements.txt
```
-->

## Running the project
To run the project, you can use Mate to run different configurations. Look at `resnet/hyperparatmers/vanilla.json` and `vit/hyperparameters/vanilla.json` for examples of configurations. Any configuration file can be selected to train. To train a model, run:
```bash
mate train {model_name} {hyperparameter_file}
```
where `{model_name}` can be anything e.g., `resnet` or `vit` and `{hyperparameter_file}` is the name of the hyperparameter file and the experiment.

## Logging
The project by default uses [Weights and Biases](https://wandb.ai/) to log the training process. You can also select any pytorch lightning loggers, e.g., `TensorBoardLogger` or `CSVLogger`. See `/vit/hyperparateres/tensorboard.json` for an example.

## Training

You can select any combination of your models with hyperparameters, for example:
```bash
mate train vit cifar100 # train vit on cifar100
mate train resnet fine_tune # fine tune a resnet trained on imagenet on cifar
mate train vit small_datasets #  model from Vision Transformer for Small-Size Datasets paper
mate train vit vanilla # original ViT paper: An Image is Worth 16x16 Words
```

You can consequently restart the training with the same configuration by running:
```bash
mate restart vit vanilla
```

## Results

You can visalize the results with any of [PyTorch Lightning⚡](https://github.com/Lightning-AI/lightning) loggers, including wandb, tensorboards, csv logger and your own custom one. By default this project uses wandb. An example of fine tuning a resnet on cifar:

<img src="/images/train_accuracy.png" alt="drawing" width="80%"/>
<img src="/images/val_accuracy.png" alt=drawing width="80%" />
<img src="/images/train_loss.png" alt="drawing" width="80%"/>
<img src="/images/val_loss.png" alt="drawing" width="80%"/>

## Experimenting and trying other models
You can try other models by changing the model in the hyperparameters or making new configuration file. Over 30 ViTs are available to experiment with. You can also fork the vit models and change the source code as you wish:
```bash
mate clone vit awesome_vit
```
Then, change the models in `project/models/awesome_vit` and keep on experimenting.

## Customizing the hyperparameters
You can customize the hyperparameters by changing the hyperparameter file. For example, you can change the  model, learning rate, batch size, optimizer, etc. this project is not limited to cifar dataset, with adding a PytorchLightningDataModule, you can train on any dataset. Optimizers, Trainers, Models and Pytorch-Lightning modules are directly created from the arguments in the configuration file and pytorch packages.

## Special thanks
Special thanks to the legend lucidrains for the [vit-pytorch](https://github.com/lucidrains/vit-pytorch) library. His licence applies to the ViT models in this project.
