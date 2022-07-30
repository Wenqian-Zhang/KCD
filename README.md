## Official Repository of KCD

```
KCD: Knowledge Walks and Textual Cues Enhanced Political Perspective Detection in News Media
Wenqian Zhang*, Shangbin Feng*, Zilong Chen*, Zhenyu Lei, Jundong Li, Minnan Luo
In *Proceedings of NAACL 2022*
```
[[<ins>Paper link</ins>](https://aclanthology.org/2022.naacl-main.304.pdf)] [[<ins>Oral Presentation Slides</ins>](https://aclanthology.org/2022.naacl-main.304.pdf)] [[<ins>Wenqian Zhang homepage</ins>](https://wenqian-zhang.github.io/)]

## Details
1. main folder consists of codes for KCD model, semmain folder indeciates codes for Semeval dataset and allmain folder indeciates codes for Allsides dataset seperately.
2. sem/Train folder is Semeval training data.
3. if you need training data for allside dataset, you can click [here](https://drive.google.com/drive/folders/1onVpTG09xYVErbidpVpaxNbEEGTduKoN?usp=sharing)
4. if you need Trained Model, you can click [here](https://drive.google.com/drive/folders/1MLtZo4KGFPqCGMmuAa8mzhr58UT_YbF6?usp=sharing)

## File structure:
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

## Dependencies
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

## How to reproduce
train the model by running
```
python Run_Model.py
```

## Citation
If this paper inspires you, please cite us!
```
@inproceedings{Zhang2022KCDKW,
  title={KCD: Knowledge Walks and Textual Cues Enhanced Political Perspective Detection in News Media},
  author={Wenqian Zhang and Shangbin Feng and Zilong Chen and Zhenyu Lei and Jundong Li and Minnan Luo},
  booktitle={NAACL},
  year={2022}
}
```

## Questions?
Feel free to open issues in this repository. You can also contact us at `2194510944@stu.xjtu.edu.cn`.
