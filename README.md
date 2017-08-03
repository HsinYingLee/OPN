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
1. [Models and Training Data](#models-and-training-data)
1. [Training](#training)
1. [Testing](#testing)

### Introduction
The Order Prediction Network (OPN) is a model that performs representation learning using unlabeled videos. Our method leverage temporal coherence as a supervisory signal by formulating representation learning as a sequence sorting task. The experimental results show that our method compares favorably against state-of-the-art methods on action recognition, image classification and object detection tasks. For more details and evaluation results, please check out our [project webpage](http://vllab1.ucmerced.edu/~hylee/OPN/) and [paper](http://vllab1.ucmerced.edu/~hylee/publication/ICCV17_OPN.pdf).

![teaser](http://vllab1.ucmerced.edu/~hylee/OPN/images/sorting.gif)

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
**Training Data**

[UCF](http://vllab1.ucmerced.edu/~hylee/OPN/results/UCF_train.mat)  /  [HMDB](http://vllab1.ucmerced.edu/~hylee/OPN/results/HMDB_train.mat) /   [UCF+HMDB+ACT](http://vllab1.ucmerced.edu/~hylee/OPN/results/UCF_HMDB_ACT.mat)

**Models**

[Model](http://vllab1.ucmerced.edu/~hylee/OPN/results/UCF_OPN.caffemodel) unsupervised trained on UCF

[Model](http://vllab1.ucmerced.edu/~hylee/OPN/results/UCFHMDBACT_nobn.caffemodel) Unsupervised trained on UCF+HMDB+ACT (for Pascal VOC 2007)

### Training
There are a few lines need to be customized in UCF_datalayers.py, including the training data location in L22-L27 and L84-87. The default setting includes all processing like channel splitting and spatial jittering, feel free to comment them out.

        $ $CAFFE_ROOT/build/tool/caffe train -solver prototxt/solver_opn.prototxt
### Testing

**Visualization**

        $ python visualize.py $MODEL $OUTPUT_FIG
        
**Action Recognition**
The testing on the UCF-101 and HMDB-51 datasets follows the testing sceme of the [original two-stream ConvNets](https://arxiv.org/pdf/1406.2199.pdf), where we sample 25 RGB frames from each video. From each of the frames we then obtain 10 inputs by cropping and flipping four corners and the center of the frame. 

**Pascal VOC 2007 Classification**
The training and testing codes are from [https://github.com/philkr/voc-classification](https://github.com/philkr/voc-classification).

**Pascal VOC 2007 Detection**
The training and testing codes are from [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
