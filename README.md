# Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images
> __Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images__  
> Xiaofei Yu, Yitong Li, Jie Ma*  
## Diffusion-RSCC
Here we provide the structure of our model:
![flow chart0520](https://github.com/Fay-Y/RSCC-Diffusion/assets/145271140/4a8c51df-f7eb-47df-ae30-d41ec38b9e9d)

## LEVIR-CC Dataset 
Download Source:
- All of the Dataset by Liu et. al:[[GitHub](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)]

## Installation and Dependencies
```python
git clone https://github.com/Fay-Y/RSCC-Diffusion
cd RSCC-Diffusion
conda create -n RSCCDiffusion_env python=3.8
conda activate RSCCDiffusion_env
pip install -r requirements.txt
```
## Training
 To train the proposed RSCC-Diffusion, run the following command:
```python
sh demo.sh
```

## Testing
 To test, evaluate and visualize on the test dataset, run the following command
```python
sh testlm.sh
```

## Visualization and Prediction samples
```python
cd result
```
In the paper, the predicted captions are saved in folder "result". Prediction results in testing with 5 Ground Truth captions are partly shown below, proving the effectiveness of our setting in the paper. 
The left part is the before image, the righ part is the after image.



#### More details will be uploaded in a few weeks.


