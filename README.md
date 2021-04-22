# Bias-Aware Loss for Training Image and Speech Quality Prediction Models from Multiple Dataset

This repository is the official implementation of following paper:  
[Gabriel Mittag, Saman Zadtootaghaj, Thilo Michael, Babak Naderi, Sebastian Möller. "Bias-Aware Loss for Training Image and Speech Quality Prediction Models from Multiple Datasets," arXiv:2104.10217 [eess.AS], 2021. (accepted at QoMEX 2021)](https://arxiv.org/abs/2104.10217)

## Requirements

To install requirements install Anaconda and then use:

```setup
conda env create -f environment.yml
```

This will create a new enviroment with the name "biasloss". Activate this enviroment to go on:

```setup2
conda activate biasloss
```

The reference images to run the synthetic experiments are included in this repo. To run the experiment with the real image quality datasets, the datasets need to be downloaded manually into a folder with following subfolders:

CSIQ: http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23 <br>
LIVE Challenge: https://live.ece.utexas.edu/research/ChallengeDB/index.html <br>
LIVE IQA R2: https://live.ece.utexas.edu/research/quality/subjective.htm <br>
LIVE MD: https://live.ece.utexas.edu/research/Quality/live_multidistortedimage.html <br>
TID 2013: http://www.ponomarenko.info/tid2013.htm 

Note however, that this is not needed to run the basic synthetic experiment.

## Experiment 1: Synthetic Image Quality Data
(In the paper this experiment is conducted on synthetic speech samples, instead of images)
To train and evaluate the first experiment on synthetic data in the paper, run this command:

```train1
python train_image_quality_synthetic.py
```


The script will plot a number of diagrams and images that can be turned off at the start of the script. For example, to turn off plotting of the results after every training epoch set "plot_every_epoch=False". Basic training options, such as learning rate, can be set in the "opts" dictionary. 

The different settings, which were analysed in the paper, can also be set there: 
* bias_fun: The polynomial function that is used to model the biases. Use either "first_order" or "third_order".
* r_th: The minimum Pearson's correlation between the predicted MOS and subjective MOS of the training dataset that needs to be achieved before the first estimation of the biases is done. Before this threshold, a vanilla MSE loss is used.
* anchor_db: Name of the dataset to which the MOS predictions are anchored to.
* mse_weight: Weight of the MSE vanilla loss that is added to the bias loss, in order to anchor the predictions.

To run the experiment without bias-aware loss set "r_th=1", in this case the biases are never updated. The blurriness sigma and the image filenames are already stored in the csv files (e.g. iqb_train.csv). 

After (and during) the training, the results of each epoch are saved in a csv file in the "results" subfolder. The csv also contains the training options that were used. In the paper, for each training setting, the training was run several times. The best epoch of each training run was then used (i.e. the column of the csv with the highest "r" value).

## Experiment 2: Real Image Quality Data

To train and evaluate the first experiment on synthetic data in the paper, run this command:

```train2
python train_image_quality_datasets.py
```

This script is similar to the first one. However, the Dataset class used is different, since it does not apply synthetic blurriness to the images. The path to the dataset folder, with the manually downloaded datasets, has to be set within the script, for example:

```
dataset_folder = 'D:/image_quality_datasets/'
```

The transformed MOS values and the filenames are already stored in "image_datasets.csv". The selection of training and validation sets are made with two lists ("train_dbs", "val_dbs") that contain the dataset names.
