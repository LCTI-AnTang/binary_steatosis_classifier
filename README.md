# Binary steatosis classifier for NAFLD

_Classification of steatosis dichotomized sets (LBUM/CRCHUM - Université de Montréal)_  

This code is related to paper   
**Comparison of Radiologists and Deep Learning for US Grading of Hepatic Steatosis**  
_Pedro Vianna, Sara-Ivana Calce, Pamela Boustros, Cassandra Larocque-Rigney, Laurent Patry-Beaudoin, Yi Hui Luo, Emre Aslan, John Marinos, Talal M. Alamri, Kim-Nhien Vu, Jessica Murphy-Lavallée, Jean-Sébastien Billiard, Emmanuel Montagnon, Hongliang Li, Samuel Kadoury, Bich N. Nguyen, Shanel Gauthier, Benjamin Therien, Irina Rish, Eugene Belilovsky, Guy Wolf, Michaël Chassé, Guy Cloutier, and An Tang_  
Radiology 2023 309:1.  

**Notes  -  
Data available in this repository is from public available sources.  
Code is presented for demonstration purposes only.**


## Introduction
This project aims to develop a binary classifier for liver steatosis in B-mode ultrasound images using deep learning methods. Liver steatosis is a common sign of non-alcoholic fatty liver disease (NAFLD). Deep learning methods applied to B-mode ultrasound images may be used as an alternative technique for screening liver steatosis.

## Getting started
The software was developed in Python (version 3.7) using the following libraries:

- Numpy (version 1.19.2)
- Pillow (version 6.2.0)
- Pandas (version 0.25.1)
- scikit-learn (version 1.0.2)
- Keras (version 2.2.4)
- Tensorflow (version 1.14.0)


## Usage
To use the software, run main.py  
`python main.py`  

with the following options:
- Dataset: csv file with filenames and steatosis scores, and definition if each image is for training or validation.  
- Architecture: VGG16, InceptionV3 and Resnet50 are implemented in arch_builder.py. Default is VGG16-dropout.  
- Transfer Learning: 'None' or path to a weights file.
- Input size: One dimension. The model will resize all images to a square shape. Default is 128.  
- Images directory: path to images directory. All images referenced in the dataset must be in this directory.  
- Task: Dichotomized steatosis setting. 0 for S0 vs >=S1, 1 for <=S1 vs >=S2, 2 for <=S2 vs S3, 9 for all tasks in sequence.  
- Mode: 'val' for training and validation, 'test' for testing. Default is 'val'. If 'test', must have a weights file given in 'transfer learning'.


## Contact information
For any questions or comments, please contact the project authors at:

Pedro Vianna: **pedro.vianna@umontreal.ca**  
An Tang: **an.tang@umontreal.ca**
