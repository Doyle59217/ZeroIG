# ZERO-IG

### Zero-Shot Illumination-Guided Joint Denoising and Adaptive Enhancement for Low-Light Images [cvpr2024]

By Yiqi Shi, Duo Liu, LiguoZhang,Ye Tian, Xuezhi Xia, Xiaojing Fu


#[[Paper]]()   [[Supplement Material]]()
# Zero-IG Framework
<img src="Figs/Fig3.png" width="900px"/> 
Note that the provided model in this code are not the model for generating results reported in the paper.

## Model Training Configuration
To train a new model, specify the dataset path in "train.py" and execute it. The trained model will be stored in the 'weights' folder, while intermediate visualization outputs will be saved in the 'results' folder.

## Requirements
* Python 3.7
* PyTorch 1.13.0
* CUDA 11.7
* Torchvision 0.14.1

## Testing
* Ensure the data is prepared and placed in the designated folder.
* Select the appropriate model for testing, which could be a model trained by yourself.
* Execute "test.py" to perform the testing.

## [VILNC Dataset]()

The Varied Indoor Luminance & Nightscapes Collection (VILNC Dataset) is a meticulously curated assembly of 500 real-world low-light images, captured with the precision of a Canon EOS 550D camera. This dataset is segmented into two main environments, comprising 460 indoor scenes and 40 outdoor landscapes. Within the indoor category, each scene is represented through a trio of images, each depicting a distinct level of dim luminance, alongside a corresponding reference image captured under normal lighting conditions. For the outdoor scenes, the dataset includes low-light photographs, each paired with its respective normal light reference image, providing a comprehensive resource for analyzing and enhancing low-light imaging techniques.

<img src="Figs/Dataset.png" width="900px"/> 



## Citation
