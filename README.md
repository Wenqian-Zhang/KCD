# KCD
Code for the NAACL 2022 paper "KCD: Knowledge Walks and Textual Cues Enhanced Political Perspective Detection in News Media"
1. main folder consists of codes for KCD model, semmain folder indeciates codes for Semeval dataset and allmain folder indeciates codes for Allsides dataset seperately.
2. sem/Train folder is Semeval training data.
3. if you need training data for allside dataset, you can click [here](https://drive.google.com/drive/folders/1onVpTG09xYVErbidpVpaxNbEEGTduKoN?usp=sharing)
4. if you need Trained Model, you can click [here](https://drive.google.com/drive/folders/1MLtZo4KGFPqCGMmuAa8mzhr58UT_YbF6?usp=sharing)


# File structure:
```
├── main               # code for KCD
      ├── allmain      # code for KCD in Allsides dataset
            ├── KSD_Dataset.py
            ├── KSD_GatedRGCN.py
            ├── Tools.py
            └── Run_Model.py      # train model
      └── semmain      # code for KCD in Semeval dataset
            ├── KSD_Dataset.py
            ├── KSD_GatedRGCN.py
            ├── Tools.py
            └── Run_Model.py      # train model
├── sem      # data for Semeval dataset
└── all      # data for Allsides dataset, you need to download from google drive
```

# Dependencies
Our code runs on the Titan X GPU with 12GB memory, with the following packages installed:
```
Python 3.8.5
torch 1.7.1
pytorch_lightning
numpy
torch_geometric
argparse
sklearn
pickle
```

# How to reproduce
train the model by running
```
python Run_Model.py
```
