# breast-cancer-classification

### Predicting invasive ductal carcinoma in tissue slices
Inasive ductal carcinoma (IDC) is - with ~ 80 % of cases - one of the most common types of breast cancer. It's malicious and able to form metastases which makes it especially dangerous.  

### What is meant by invasive ductal carcinoma?

<img align="left" src="https://upload.wikimedia.org/wikipedia/commons/4/47/Lobules_and_ducts_of_the_breast.jpg" width="240" height="240">
<br/>

This illustration created Mikael Häggström shows the anatomy of a healthy breast. One can see the lobules, the glands that can produce milk which flews through the milk ducts. Ductal carcinoma starts to develop in the ducts whereas lobular carcinoma has its origin in the lobules. Invasive carcinoma is able to leave its initial tissue compartment and can form metastases.  

<br/>
<br/>
<br/>
<br/>




### The Dataset
We’ll use the IDC_regular dataset ([the breast cancer histology image dataset!](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/)) from Kaggle. This dataset holds 2,77,524 patches of size 50×50 extracted from 162 whole mount slide images of breast cancer specimens scanned at 40x. Of these, 1,98,738 test negative and 78,786 test positive with IDC. 
Please try to run [data_visualization!](data_visualization.ipynb) first to get better idea of the data.  

#### Please run [build_dataset](build_dataset.py) to prepare data before training.

<br/>

!Many models are added inside [customModel.py](model/customModel.py). Only tested onces so far are MobileNetV2 and custom model.
!MobileNetV2 first is transferd learnt using imagenet weights, and later finetuned. (please check the accuracy and loss plotted for  [transfer learnt](mobilenet_frozen_network.png) and for [fine tunned](mobilenet_unfrozen_network.png) )
!MobileNetV2 trained model weights are already uploaded.
