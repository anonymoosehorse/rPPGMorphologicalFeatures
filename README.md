# Prediction of morphological blood volume pulse features using DeepNetworks

---
 > This repository is currently under construction!

## Datasets

Currently two datasets are used for testing. 
- VIPL-HR: https://arxiv.org/pdf/1810.04927.pdf
- In-House-dataset: available on request

## Data preprocessing

To avoid extra work down the line rename the Videos in the VIPL dataset to match their folder structure e.g. `.../p1/v1/source1/video.avi` should become `.../p1/v1/source1/p1_v1_s1.avi`

### Trace generation

For trace generation use a face detector or choice, select a face region and average the region for each of the three color channels. The trace files should be .json files with the same filename as the video files they were extracted from (e.g. `p1_v1_s1.json`). 

The .json file must contain a dictionary with the keys ["R", "G" and "B"] each containing a list of average color value for the corresponding color channels.

See [this](examples/01_01.json) as an example.

### Adjusting configuration

Use the [dataset_config.yaml](dataset_config.yaml) file to point to the extracted traces files and the ground truth files.

Use the [preprocess_config.yaml](preprocess_config.yaml) to determine which dataset to preprocess. 

### Run Preprocessing 

Run the [preprocess.py](/src/preprocess.py) script to preprocess the datasets for training:

```bash
python ./src/preprocess.py
```

<!-- ## IBIS trace generation -->

## Training a network

Use the [config.yaml](config.yaml) file to set the settings for the analysis. 

Use the [lightning_main.py](/src/lightning_main.py) script to train on the preprocessed data:

```bash
python ./src/lightning_main.py
```