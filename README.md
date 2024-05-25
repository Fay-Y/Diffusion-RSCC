# Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images
> __Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images__  
> Xiaofei Yu, Yitong Li, Jie Ma*  [[paper](https://arxiv.org/abs/2405.12875)]
## Diffusion-RSCC
Here we provide the structure of our model:

![flow chart0521](https://github.com/Fay-Y/Diffusion-RSCC/assets/145271140/a8b7e4a4-0317-46c1-8e04-8b3aadc569fc)

## LEVIR-CC Dataset 
Download Source:
-Thanks for the Dataset by Liu et. al:[[GitHub](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)].
Put the content of downloaded dataset under the folder 'data'
```python
path to ./data:
                ├─LevirCCcaptions.json
                ├─images
                  ├─train
                  │  ├─A
                  │  ├─B
                  ├─val
                  │  ├─A
                  │  ├─B
                  ├─test
                  │  ├─A
                  │  ├─B
```

## Installation and Dependencies
```python
git clone https://github.com/Fay-Y/Diffusion-RSCC
cd Diffusion-RSCC
conda create -n DiffusionRSCC_env python=3.8
conda activate DiffusionRSCC_env
pip install -r requirements.txt
```
## Preparation
Preprocess the raw captions and image pairs:
```python
python word_encode.py
python img_preprocess.py
```

## Training
 To train the proposed Diffusion-RSCC, run the following command:
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
![github](https://github.com/Fay-Y/Diffusion-RSCC/assets/145271140/16ca7a45-a4bd-4aff-8878-26cf48d8caf7)





