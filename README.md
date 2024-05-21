# Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images
> __Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images__  
> Xiaofei Yu, Yitong Li, Jie Ma*  
## Diffusion-RSCC
Here we provide the structure of our model:

![flow chart0521](https://github.com/Fay-Y/Diffusion-RSCC/assets/145271140/a8b7e4a4-0317-46c1-8e04-8b3aadc569fc)

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

## Visualization
```python
cd result
```
In the paper, the predicted captions are saved in folder "result". 
## Prediction samples
Prediction results in test set with 5 Ground Truth captions are partly shown below, proving the effectiveness of our model. 
For each image pair, the left part is the before image, the righ part is the after image.
![1](https://github.com/Fay-Y/Diffusion-RSCC/assets/145271140/048c0dcc-cdae-423d-b021-a2ba1a4a1d9d)

![2](https://github.com/Fay-Y/Diffusion-RSCC/assets/145271140/cefc4fa0-d1f3-47cf-ae29-ec75595e26d6)

![3](https://github.com/Fay-Y/Diffusion-RSCC/assets/145271140/9920745a-9363-47ba-9289-efa3d9aa572b)

#### More details will be uploaded in a few weeks.


