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
### Installation and Dependencies
```python
git clone https://github.com/Fay-Y/RSCC-Diffusion
cd RSCC-Diffusion
conda create -n RSCCDiffusion_env python=3.8
conda activate RSCCDiffusion_env
pip install -r requirements.txt

### Training
 To train the proposed method, run the following commands:
'''
sh demo.sh
'''

#### Testing
 To test evaluate and visualize on the test dataset, run the following command
'''
sh testlm.sh
'''
