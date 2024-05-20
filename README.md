# RSCC-Diffusion
> __RSCC-Diffusion: Bridging Text and Bi-temporal Images for Change Captioning in Remote Sensing__  
> _Xiaofei Yu, Jie Ma*, Yitong Li_  

## LEVIR-CC Dataset 
Download Source:
- All of the Dataset by Liu et. al:[[GitHub](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset))]
```python
path to LEVIR_CC_dataset:
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
git clone https://github.com/Fay-Y/RSCC-Diffusion
cd RSCC-Diffusion
conda create -n RSCCDiffusion_env python=3.8
conda activate RSCCDiffusion_env
pip install -r requirements.txt
```
## Training
 To train the proposed method, run the following commands:
```python
sh demo.sh
```

## Testing
 To test evaluate and visualize on the test dataset, run the following command
```python
sh testlm.sh
```

## Visualization
```python
cd result
```
In the paper, the predicted captions are saved in file "result". Prediction results in testing with random one of Ground Truth captions are 
partly shown below, proving the effectiveness of our setting in the paper. 
![test_000242](https://github.com/Fay-Y/RSCC-Diffusion/assets/145271140/b4d623c1-3f7b-436d-93ac-7862a018b051)
![test_001127](https://github.com/Fay-Y/RSCC-Diffusion/assets/145271140/ae323b5f-0fb9-457a-b2b0-8bedc15c93a4)
![test_001319](https://github.com/Fay-Y/RSCC-Diffusion/assets/145271140/50bb377c-947a-43ed-b91a-dc853d578df4)
![test_001960](https://github.com/Fay-Y/RSCC-Diffusion/assets/145271140/142740a0-239e-42cd-9020-45a2b398efa3)


####More details will be upload in a few weeks


