# Variational Autoencoder collection
- VAE
- $\beta$-VAE
- $\sigma$-VAE
- 2-stage VAE

## Setup
Tested with Python 3.8.5 on Ubuntu 20.04.  

Make virtual environment:
```
pip install pip --upgrade
pip install virtualenv
python3 -m venv env
```
Activate env:
```
source env/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```

Implemented datasets:
```
vaes/data_utils/higgs
vaes/data_utils/mnist
```

## Logging
Runs are saved in ```mlruns/``` and can be accesed with:
```
mlflow ui
```

## Training
Configure hydra conf files and run directly from model scripts.

## TODO
- convolutional neural networks
- more datasets
- other VAE models
