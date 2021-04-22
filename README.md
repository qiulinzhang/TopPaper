- [TopPaper](#toppaper)
    + [Classic Papers for Beginners, Hot Orientation for Researchers, and Impact Scope for Authors.](#classic-papers-for-beginners--hot-orientation-for-researchers--and-impact-scope-for-authors)
  * [0. Traditional Methods](#0-traditional-methods)
  * [1. CNN [Convolutional Neural Network]](#1-cnn--convolutional-neural-network-)
    + [1.1 Image Classification](#11-image-classification)
      - [1.1.1 Architecture](#111-architecture)
      - [1.1.2 Dataset, Augmentation, Trick](#112-dataset--augmentation--trick)
    + [1.2 Object Detection](#12-object-detection)
    + [1.3 Object Segmentation](#13-object-segmentation)
    + [1.4 Re_ID [Person Re-Identification]](#14-re-id--person-re-identification-)
    + [1.5 OCR [Optical Character Recognition]](#15-ocr--optical-character-recognition-)
    + [1.6 Face Recognition](#16-face-recognition)
    + [1.7 NAS [Neural Architecture Search]](#17-nas--neural-architecture-search-)
    + [1.8 Image Super_Resolution](#18-image-super-resolution)
    + [1.9 Image Denoising](#19-image-denoising)
    + [1.10 Model Compression, Pruning, Quantization, Knowledge Distillation](#110-model-compression--pruning--quantization--knowledge-distillation)
  * [2. Transformer in Vision](#2-transformer-in-vision)
  * [3. Transformer and Self-Attention in NLP](#3-transformer-and-self-attention-in-nlp)
  * [4. Others](#4-others)
  * [Acknowledgement](#acknowledgement)

# TopPaper
### Classic Papers for Beginners, Hot Orientation for Researchers, and Impact Scope for Authors.
There have been billions of academic papers around the world. However, maybe only 0.0...01\% among them are valuable or are worth reading. Since our limited life has never been forever, **TopPaper** provide a **Top Academic Paper Chart** for beginners and reseachers to take one step faster.

Welcome to contribute more subject or valuable (at least you think) papers. Please feel free to pull requests or open an issue. 

---

## 0. Traditional Methods
| Abbreviation |                                                          Paper                                                         | Cited by | Journal | Year |   1st Author  |         1st Affiliation        |
|:------------:|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:-------------:|:------------------------------:|
| SIFT         | [Object Recognition from Local Scale-Invariant   Features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)              |   20 K   | ICCV    | 1999 | David G. Lowe | University of British Columbia |
| HOG          | [Histograms of Oriented Gradients for Human   Detection](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) |   35 K   | CVPR    | 2005 | Navneet Dalal | inrialpes                      |
| SURF         | [SURF: Speeded Up Robust   Features](https://people.ee.ethz.ch/~surf/eccv06.pdf)                                       |   18 K   | ECCV    | 2006 | Herbert Bay   | ETH Zurich                     |
......

## 1. CNN [Convolutional Neural Network]
### 1.1 Image Classification
#### 1.1.1 Architecture
|     Abbreviation |                                                                                 Paper                                                                            |    Cited By |           Journal       |     Year |       1st Author  |           1st Affiliation      |
|------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|:-----------------------:|:--------:|:-----------------:|:------------------------------:|
| LeNet            | [Backpropagation applied to   handwritten zip code   recognition](http://www.iro.umontreal.ca/~lisa/bib/pub_subject/finance/pointeurs/lecun-98.pdf)              |    8.3 K    |    Neural Computation   |   1989   | Yann Lecun        |     AT&T Bell Laboratories     |
| LeNet            | [Gradient-based learning applied to   document   recognition](https://mila.quebec/wp-content/uploads/2019/08/Gradient.pdf)                                       |    35 K   | Proceedings of the IEEE |   1998   | Yann Lecun        |     AT&T Research Laboratories |
| ImageNet            | [ImageNet: A large-scale hierarchical image database](http://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)                                       |    26 K   | CVPR |   2009   | Jia Dengn        |     Princeton University |
| AlexNet          | [ImageNet   Classification with Deep Convolutional      Neural   Networks](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf) |     79 K    |           NIPS          |   2012   | Alex Krizhevsky   | University of Toronto          |
| ZFNet            | [Visualizing and Understanding   Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)                                                                    |     11 K    |           ECCV          |   2014   | Matthew D Zeiler  | New York University            |
| VGGNet           | [Very Deep Convolutional Networks   for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)                                                      |     55 K    |           ICLR          |   2015   | Karen Simonyan    | Oxford                         |
| GoogLeNet        | [Going Deeper with   Convolutions](https://arxiv.org/pdf/1409.4842.pdf)                                                                                          |     29 K    |           CVPR          |   2015   | Christian Szegedy | Google                         |
| GoogLeNet_v2_v3  | [Rethinking the Inception   Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)                                                              |     12 K    |           CVPR          |   2016   | Christian Szegedy | Google                         |
| ResNet           | [Deep Residual Learning for Image   Recognition](https://arxiv.org/pdf/1512.03385.pdf)                                                                           |     74 K    |           CVPR          |   2016   | Kaiming He        | MSRA                           |
| DenseNet         | [Densely Connected Convolutional   Networks](https://arxiv.org/pdf/1608.06993.pdf)                                                                               |     15 K    |           CVPR          |   2017   | Gao Huang         | Cornell University             |
| ResNeXt         | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)                                                                               |     3.9 K    |           CVPR          |   2017   | Saining Xie        | UC San Diego             |
| MobileNet        | [MobileNets: Efficient   Convolutional Neural Networks for Mobile Vision   Applications](https://arxiv.org/pdf/1704.04861.pdf)                                   |    7.7 K    |          arXiv          |   2017   | Andrew G. Howard  | Google                         |
| SENet            | [Squeeze-and-Excitation   Networks](https://arxiv.org/pdf/1709.01507.pdf)                                                                                        |    6.3 K    |           CVPR          |   2018   | Jie Hu            | Momenta                        |
| MobileNet_v2     | [MobileNetV2: Inverted Residuals   and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)                                                                 |    4.4 K    |           CVPR          |   2018   | Mark Sandler      | Google                         |
| ShuffleNet       | [ShuffleNet: An Extremely Efficient Convolutional Neural   Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)                                     |    2.3 K    |           CVPR          |   2018   | Xiangyu Zhang     | Megvii                         |
| ShuffleNet V2    | [ShuffleNet V2: Practical Guidelines for Efficient CNN   Architecture Design](https://arxiv.org/pdf/1807.11164.pdf)                                              |    1.3 K    |           ECCV          |   2018   | Ningning Ma       | Megvii                         |
| MobileNet_v3     | [Searching for   MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)                                                                                              |    0.6 K    |           ICCV          |   2019   | Andrew Howard     | Google                         |
| EfficientNet     | [EfficientNet: Rethinking Model   Scaling for Convolutional Neural   Networks](https://arxiv.org/pdf/1905.11946.pdf)                                             |    1.9 K    |           ICML          |   2019   | Mingxing Tan      | Google                         |
| GhostNet         | [GhostNet: More Features from Cheap   Operations](https://arxiv.org/pdf/1911.11907.pdf)                                                                          |    0.1 K    |           CVPR          |   2020   | Kai Han           | Huawei                         |
| Res2Net          | [Res2Net: A New Multi-scale Backbone   Architecture](https://arxiv.org/pdf/1904.01169.pdf)                                                                       |    0.2 K    |          TPAMI          |   2021   | Shang-Hua Gao     | Nankai University              |

#### 1.1.2 Dataset, Augmentation, Trick
| Abbreviation |                                                         Paper                                                         | Cited By | Journal | Year |   1st Author  |      1st Affiliation     |
|:------------:|:---------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:-------------:|:------------------------:|
|-|[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)|361|CVPR|2019|Tong He|Amazon|
|-|[Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423)|122|NeurIPS|2019|Hugo Touvron|FAIR|
|Auto-Augment|[AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)|487|CVPR|2019|Ekin D. Cubuk|Google|
|-|[Fixing the train-test resolution discrepancy: FixEfficientNet](https://arxiv.org/abs/2003.08237)|53|Arxiv|2020|Hugo Touvron|FAIR|
### 1.2 Object Detection
| Abbreviation |                                                         Paper                                                         | Cited By | Journal | Year |   1st Author  |      1st Affiliation     |
|:------------:|:---------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:-------------:|:------------------------:|
| RCNN         | [Rich feature hierarchies for accurate object detection and semantic   segmentation](https://arxiv.org/abs/1311.2524) |   17 K   |   CVPR  | 2014 | Ross Girshick | Berkeley                 |
| Fast RCNN    | [Fast R-CNN](https://arxiv.org/abs/1504.08083)                                                                        |   14 K   |   ICCV  | 2015 | Ross Girshick | Microsoft Research       |
| Faster RCNN  | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal   Networks](https://arxiv.org/abs/1506.01497)  |   20 K   |   NIPS  | 2015 | Shaoqing Ren  | USTC, MSRA               |
| SSD          | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)                                                |   13 K   |   ECCV  | 2016 | Wei Liu       | UNC                      |
| YOLO         | [You Only Look Once: Unified, Real-Time Object   Detection](https://arxiv.org/abs/1506.02640)                         |   15 K   |   CVPR  | 2016 | Joseph Redmon | University of Washington |
| Mask RCNN    | [Mask R-CNN](https://arxiv.org/abs/1703.06870)                                                                        |   10 K   |   ICCV  | 2017 | Kaiming He    | FAIR                     |
| DSSD         | [DSSD : Deconvolutional Single Shot   Detector](https://arxiv.org/abs/1701.06659)                                     |   1.0 K  |   CVPR  | 2017 | Cheng-Yang Fu | UNC                      |
| YOLO9000     | [YOLO9000: Better, Faster, Stronger.](https://arxiv.org/abs/1612.08242)                                               |   7.7 K  |   CVPR  | 2017 | Joseph Redmon | University of Washington |
| FPN          | [Feature Pyramid Networks for Object   Detection](https://arxiv.org/abs/1612.03144)                                   |   6.7 K  |   CVPR  | 2017 | Tsung-Yi Lin  | FAIR                     |
| Focal Loss   | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)                                             |   6.7 K  |   ICCV  | 2017 | Tsung-Yi Lin  | FAIR                     |
|Deformable Conv|[Deformable Convolutional Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)|1.6 K|ICCV|2017|Jifeng Dai|MSRA|
| YOLO V3      | [Yolov3: An incremental improvement](https://arxiv.org/abs/1804.02767)                                                |   6.9 K  |   CVPR  | 2018 | Joseph Redmon | University of Washington |
| ATSS         | [Bridging the Gap Between Anchor-based and Anchor-free Detection via   Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424)          |   0.1 K  |   CVPR  | 2020 | Shifeng Zhang | CASIA                    |
| EfficientDet | [EfficientDet: Scalable and Efficient Object   Detection](https://arxiv.org/abs/1911.09070)                           |   0.3 K  |   CVPR  | 2020 | Mingxing Tan  | Google                   |

### 1.3 Object Segmentation
|     Abbreviation |                                                                            Paper                                                                        |    Cited By |     Journal |     Year |       1st Author  |         1st Affiliation    |
|------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|:-----------:|:--------:|:-----------------:|:--------------------------:|
| FCN              | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)                                                             | 22 K        | CVPR        | 2015     | Jonathan Long     | UC Berkeley                |   |
| DeepLab          | [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected   CRFs](https://arxiv.org/abs/1606.00915) | 7.4 K       | ICLR        | 2015     | Liang-Chieh Chen  | Google                     |   |
| Unet             | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)                                                   | 24 K        | MICCAI      | 2015     | Olaf Ronneberger  | University of Freiburg     |   |
| -                | [Learning to Segment Object Candidates](https://arxiv.org/abs/1506.06204)                                                                               | 0.6 K       | NIPS        | 2015     | Pedro O. Pinheiro | FAIR                       |   |
| Dilated Conv     | [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)                                                           | 4.5 K       | ICLR        | 2016     | Fisher Y          | Princeton University       |   |
| -                | [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719)                             | 0.7 K       | CVPR        | 2017     | Chao Peng         | Tsinghua                   |   |
| RefineNet        | [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)                               | 1.6 K       | CVPR        | 2017     | Guosheng Lin      | The University of Adelaide |   |
### 1.4 Re_ID [Person Re-Identification]
### 1.5 OCR [Optical Character Recognition]
| Abbreviation |                                                                              Paper                                                                              | Cited by | Journal | Year |  1st Author | 1st Affiliation |
|:------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:-----------:|:---------------:|
| CTC          | [Connectionist   temporal classifaction: labelling unsegmented sequence data with recurrent   neural network](https://www.cs.toronto.edu/~graves/icml_2006.pdf) |   2.9 K  | ICML    | 2006 | Alex Graves | IDSIA           |
### 1.6 Face Recognition
|   Abbreviation  |                                                                                        Paper                                                                                       | Cited by |          Journal          | Year |   1st Author   |         1st Affiliation         |
|:---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:-------------------------:|:----:|:--------------:|:-------------------------------:|
| DeepFace        | [DeepFace: Closing the Gap to Human-Level Performance in Face   Verification](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)                                 |   5.3 K  | CVPR                      | 2014 | Yaniv Taigman  | FAIR                            |
| DeepID v1       | [Deep Learning Face Representation from Predicting 10,000   Classes](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf) |   1.8 K  | CVPR                      | 2014 | Yi Sun         | CUHK                            |
| DeepID v2       | [Deep Learning Face Representation by Joint   Identification-Verification](https://arxiv.org/abs/1406.4773)                                                                        |   1.9 K  | NIPS                      | 2014 | Yi Sun         | CUHK                            |
| FaceNet         | [FaceNet: A Unified Embedding for Face Recognition and   Clustering](https://arxiv.org/abs/1503.03832)                                                                             |   7.4 K  | CVPR                      | 2015 | Florian Schrof | Google                          |
| Center Loss     | [A Discriminative Feature   Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)                                                                                                            |   2.1 K  | ECCV                      | 2016 | Yandong Wen    | CMU                             |
| ArcFaxe         | [ArcFace: Additive Angular Margin Loss for Deep Face   Recognition](https://arxiv.org/abs/1801.07698)                                                                              |   1.3 K  | CVPR                      | 2017 | Jiankang Deng  | Imperial College London         |
| SphereFace      | [SphereFace: Deep Hypersphere Embedding for Face   Recognition](https://arxiv.org/abs/1704.08063)                                                                                  |   1.3 K  | CVPR                      | 2017 | Weiyang Liu    | Georgia Institute of Technology |
| CosFace         | [CosFace: Large Margin Cosine Loss for Deep Face   Recognition](https://arxiv.org/abs/1801.09414)                                                                                  |   0.8 K  | CVPR                      | 2018 | Hao Wang       | Tecent                          |
| AM-Softmax Loss | [Additive Margin Softmax for Face   Verification](https://arxiv.org/abs/1801.05599)                                                                                                |   0.5 K  | Signal Processing Letters | 2018 | Feng Wang      | UESTC                           |
### 1.7 NAS [Neural Architecture Search]
| Abbreviation |                                              Paper                                             | Cited By | Journal | Year |  1st Author | 1st Affiliation |
|--------------|:----------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:-----------:|:---------------:|
| Darts        | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)                  |   1.3 K  |   ICLR  | 2019 | Hanxiao Liu | CMU             |
| -            | [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)     |   2.5 K  |   ICLR  | 2017 | Barret Zoph | Google          |
| -            | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) |   1.2 K  |   ICML  | 2018 | Hieu Pham   | Google          |
| -            | [SNAS: Stochastic Neural Architecture Search](https://arxiv.org/abs/1812.09926)                |   0.3 K  |   ICLR  | 2019 | Sirui Xie   | SenseTime       |
|PC-Darts|[PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search](https://arxiv.org/abs/1907.05737)| 159|ICLR|2020|Yuhui Xu|Huawei|
### 1.8 Image Super_Resolution
| Abbreviation |                                                                                   Paper                                                                                   | Cited By | Journal | Year |    1st Author   |      1st Affiliation      |
|--------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:---------------:|:-------------------------:|
| SRCNN        | [Image Super-Resolution Using Deep Convolutional   Networks](https://arxiv.org/abs/1501.00092)                                                                            |   4.1 K  |   ECCV  | 2014 | Chao Dong       | CUHK                      |
| ESPCN        | [Real-Time Single Image and Video Super-Resolution Using an Efficient   Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)                         |   2.4 K  |   CVPR  | 2016 | Wenzhe Shi      | Twitter                   |
| FSRCNN       | [Accelerating the Super-Resolution Convolutional Neural   Network](https://arxiv.org/abs/1608.00367)                                                                      |   1.3 K  |   ECCV  | 2016 | Chao Dong       | CUHK                      |
| VDSR         | [Accurate Image Super-Resolution Using Very Deep Convolutional   Networks](https://arxiv.org/abs/1511.04587)                                                              |   3.5 K  |   CVPR  | 2016 | Jiwon Kim       | Seoul National University |
| DRCN         | [Deeply-Recursive Convolutional Network for Image   Super-Resolution](https://arxiv.org/abs/1511.04491)                                                                   |   1.4 K  |   CVPR  | 2016 | Jiwon Kim       | Seoul National University |
| EDSR         | [Enhanced Deep Residual Networks for Single Image   Super-Resolution](https://arxiv.org/abs/1707.02921)                                                                   |   2.0 K  |  CVPRW  | 2017 | Bee Lim         | Seoul National University |
| DRRN         | [Image Super-Resolution via Deep Recursive Residual   Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf) |   1.0 K  |   CVPR  | 2017 | Ying Tai        | NJUST                     |
| SRDenseNet   | [Image Super-Resolution Using Dense Skip   Connections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)     |   0.5 K  |   ICCV  | 2017 | Tong Tong       | Imperial Vision           |
| SRGAN        | [Photo-Realistic Single Image Super-Resolution Using a Generative   Adversarial Network](https://arxiv.org/abs/1609.04802)                                                |   5.3 K  |   CVPR  | 2017 | Christian Ledig | Twitter                   |
|LapSRN|[Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/abs/1704.03915)|1.1 K|CVPR|2017|Wei-Sheng Lai|1University of California|
|RDN|[Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)|1.1 K|CVPR|2018|Yulun Zhang|Northeastern University|
|DBPN|[Deep Back-Projection Networks For Super-Resolution](https://arxiv.org/abs/1803.02735)|0.6 K|CVPR|2018|Muhammad Haris|Toyota Technological Institute|
|RCAN|[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)|1.0 K|ECCV|2018|Yulun Zhang|Northeastern University|

### 1.9 Image Denoising
| Abbreviation |                                                           Paper                                                          | Cited By | Journal | Year | 1st Author | 1st Affiliation |
|:------------:|:------------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:----------:|:---------------:|
| CBDNet       | [Toward Convolutional Blind Denoising of Real   Photographs](https://arxiv.org/abs/1807.04686)                           |   0.2 K  | CVPR    | 2019 | Shi Guo    | HIT             |
| -            | [Learning Deep CNN Denoiser Prior for Image   Restoration](https://arxiv.org/abs/1704.03264)                             |   0.8 K  | CVPR    | 2017 | Kai Zhang  | HIT             |
| CnDNN        | [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image   Denoising](https://arxiv.org/abs/1608.03981)      |   2.9 K  | TIP     | 2017 | Kai Zhang  | HIT             |
| FFDNet       | [FFDNet: Toward a fast and flexible solution for CNN based image   denoising](https://arxiv.org/abs/1710.04026)          |   0.6 K  | TIP     | 2018 | Kai Zhang  | HIT             |
| SRMD         | [Learning a Single Convolutional Super-Resolution Network for Multiple   Degradations](https://arxiv.org/abs/1712.06116) |   0.3 K  | CVPR    | 2018 | Kai Zhang  | HIT             |
|RIDNet|[Real Image Denoising with Feature Attention](https://arxiv.org/abs/1904.07396)]|87|ICCV|2019|Saeed Anwar|CSIRO|
|CycleISP|[CycleISP: Real Image Restoration via Improved Data Synthesis](https://arxiv.org/abs/2003.07761)|28|CVPR|2020|Syed Waqas Zamir|UAE|
|AINDNet|[Transfer Learning from Synthetic to Real-Noise Denoising with Adaptive Instance Normalization](https://arxiv.org/abs/2002.11244)|14|CVPR|2020|Yoonsik Kim|Seoul National University|

### 1.10 Model Compression, Pruning, Quantization, Knowledge Distillation
| Abbreviation |                                                           Paper                                                          | Cited By | Journal | Year | 1st Author | 1st Affiliation |
|:------------:|:------------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:----------:|:---------------:|
| KD       | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)                           |   5.8 K  | NIPS-W    | 2014 | Geoffrey Hinton    | Google             |
|DeepCompression|[Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)|4.9K|ICLR|2016|Song Han|Stanford|
|Fixed Point Quant|[Fixed point quantization of deep convolutional networks](https://openreview.net/pdf?id=yovBjmpo1ur682gwszM7)|0.5 K|ICLR-W|2016|Darryl D. Lin|Qualcomm|
|DoReFa|[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)|1.1 K|CVPR|2016|Shuchang Zhou|Megvii|
|Fake Quant|[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)|0.8 K|CVPR|2018|Benoit Jacob|Google|
|Once for all|[Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791)|0.1 K| ICLR|2020|Han Cai|MIT|
---

## 2. Transformer in Vision
|   Abbreviation  |                                                      Paper                                                     | Cited by | Journal | Year |     1st Author     | 1st Affiliation |
|:---------------:|:--------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:------------------:|:---------------:|
|Image Transformer|[Image Transformer](https://arxiv.org/abs/1802.05751)|337|ICML|2018|Niki Parmar|Google|
|-|[Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925)|191|ICCV|2019|Irwan Bello|Google|
| DETR            | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)                              |    252   |   ECCV  | 2020 | Nicolas Carion     | Facebook AI     |
|i-GPT|[Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)|38|ICML|2020|Mark Chen|OpenAI|
| Deformable DETR | [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)   |    12    |   ICLR  | 2021 | Xizhou Zhu         | SenseTime       |
|-|[Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)|57|Arxiv|2020|Hugo Touvron|FAIR|
| ViT             | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) |    175   |   ICLR  | 2021 | Alexey Dosovitskiy | Google          |
| IPT             | [Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364)                                   |    16    |   CVPR  | 2021 | Hanting Chen       | Huawei Noah          |
|-|[A Survey on Visual Transformer](https://arxiv.org/abs/2012.12556)|12|Arxiv|2021|Kai Han|Huawei Noah|
| TNT             | [Transformer in Transformer](https://arxiv.org/abs/2103.00112)                                                 |     8    |  Arxiv  | 2021 | Kai Han            | Huawei Noah         |
......

---
## 3. Transformer and Self-Attention in NLP
|   Abbreviation  |                                                      Paper                                                     | Cited by | Journal | Year |     1st Author     | 1st Affiliation |
|:---------------:|:--------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|:----:|:------------------:|:---------------:|
|Transformer|[Attention Is All You Need](https://arxiv.org/abs/1706.03762)|19 K|NIPS|2017|Ashish Vaswani|Google|
|-|[Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)|0.5 K|NAACL|2018|Peter Shaw|Google|
|Bert|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)|17 K|NAACL|2019|Jacob Devlin|Google|

---
## 4. Others
......


## Acknowledgement 
Thanks for the materias and help from Aidong Men, Bo Yang, Zhuqing Jiang, Qishuo Lu, Zhengxin Zeng, Jia'nan Han, Pengliang Tang, Yiyun Zhao, Xian Zhang ......
