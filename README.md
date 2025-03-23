# Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images
> __Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images__  
> Xiaofei Yu, Yitong Li, Jie Ma*， Chang Li*, Hanlin Wu  [[paper](https://arxiv.org/abs/2405.12875)]

##  Model Architecture
The proposed Diffusion-RSCC consists of:
- A **forward diffusion process** that adds noise to caption embeddings until they resemble Gaussian noise.
- A **reverse denoising process** using a specially designed **Condition Denoiser**:
  - **Feature Extractor**: Pretrained ResNet101 to extract features from bi-temporal images.
  - **Cross-Mode Fusion (CMF)**: Integrates visual and textual modalities for precise alignment.
  - **Stacking Self-Attention (SSA)**: Refines cross-modal information for accurate conditional mean estimation.
- The denoised latent vectors are converted into natural language captions.

![flowchart](https://github.com/user-attachments/assets/2bc99e09-b23b-416c-aa63-044a059be52f)

### Datasets
#### LEVIR-CC
- A large-scale RSICC dataset with 10,077 bi-temporal image pairs and 50,385 captions.
- Covers multiple semantic change types: buildings, roads, vegetation, parking lots, water.
- Resized images: 256×256.

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
#### DUBAI-CC
- Contains 500 urban area image pairs with 2500 annotations for changes in roads, buildings, lakes, etc.
- Resized into 256×256 in Diffusion-RSCC.
- Focuses on urbanization and land cover changes over 10 years.


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
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/eaf7ba0c-1a4d-44cd-9d11-84bfda0058ab" alt="compare2" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/b61bad59-afd0-4313-9b97-d7ab859222eb" alt="compare1" width="500"/></td>
  </tr>
</table>

## TODO
- [ ] Release training logs and checkpoints
- [ ] Support more RSICC datasets






