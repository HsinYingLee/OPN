# Unsupervised Representation Learning by Sorting Sequence (ICCV 2017)
[Hsin-Ying Lee](http://vllab1.ucmerced.edu/~hylee/),
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/),
[Maneesh Kumar Singh](https://scholar.google.com/citations?user=hdQhiFgAAAAJ),
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

IEEE International Conference on Computer Vision, ICCV 2017

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements)
1. [Models and Training Data](#models_and_training_data)
1. [Training](#training)
1. [Testing](#testing)

### Introduction
The Order Prediction Network (OPN) is a model that performs representation learning using unlabeled videos. Our method leverage temporal coherence as a supervisory signal by formulating representation learning as a sequence sorting task. The experimental results show that our method compare favorably against state-of-the-art methods on action recognition, image classification and object detection tasks. For more details and evaluation results, please check out our [project webpage](http://vllab1.ucmerced.edu/~hylee/OPN/) and [paper](http://vllab1.ucmerced.edu/~hylee/publication/ICCV17_OPN.pdf).

### Citation
If you find the code and data useful in your research, please cite:
    
    @inproceeding{OPN,
        author    = {Lee, Hsin-Ying and Huang, Jia-Bin and Singh, Maneesh Kumar and Yang, Ming-Hsuan}, 
        title     = {Unsupervised Representation Learning by Sorting Sequence}, 
        booktitle = {IEEE International Conference on Computer Vision},
        year      = {2017}
    }

### Requirements
Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
Note that the Caffe fork needs to support Batch Normalization to run our code.

### Models and Training Data

### Training

### Testing
