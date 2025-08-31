# Face Mask Classifier (FastAI)

A deep learning project that classifies whether people are:

- with_mask  
- without_mask  
- mask_weared_incorrect  

Built end-to-end in under ~5 hours as part of learning [FastAI’s course](https://course.fast.ai/), Chapter 2.

---

## Project Overview

**Goal**: Build a 3-class image classifier with >70% validation accuracy and deploy it with a local Gradio UI.  

**Definition of Done**:
- 3-class classifier trained on Kaggle’s [Face Mask Detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)  
- Clean preprocessing pipeline (`labels.csv` with risk-based priority rules)  
- Training with FastAI’s `vision_learner` (ResNet34 backbone)  
- Model achieves ~74% validation accuracy  
- Confusion matrix and misclassification analysis  
- Gradio app for quick demo  
- Exported `.pkl` model for reuse  

---

## Project Structure

```
Project1-Face-Mask-Classifier/
│
├── data/
│   ├── annotations/           # XML labels (not committed, from Kaggle)
│   ├── images/               # Images (not committed, from Kaggle)
│   └── labels.csv            # collapsed labels (source of truth)
│
├── notebooks/
│   └── FaceMaskClassifier.ipynb  # main notebook
│
├── models/
│   └── mask_classifier.pkl   # exported fastai model
│
├── app.py                    # Gradio demo script
└── README.md                 # project documentation
```

---

## Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/YOURUSERNAME/face-mask-classifier.git
   cd face-mask-classifier
    ```
2. Create and activate conda environment:
  ```bash
  conda create -n fastai python=3.11 -y
  conda activate fastai
  ```

3. Install dependencies:
  ``` bash
  pip install fastai gradio
```
4. Download dataset:
  ### From [Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?resource=download)
* Unzip into `data/images/` and `data/annotations/`.

# Usage

## Jupyter Notebook Training

```bash
jupyter lab
```

Open `notebooks/FaceMaskClassifier.ipynb` and run all cells to retrain or explore.

## Run Gradio App

```bash
python app.py
```

Upload an image and get predictions.

## Results

* Model: ResNet34 (FastAI `vision_learner`)
* Validation Accuracy: ~74%
* Confusion Matrix: shows most errors are between `with_mask` and `without_mask` in crowded scenes.

## Next Steps

* Improve accuracy with data augmentation
* Try ResNet50 / EfficientNet
* Deploy to Hugging Face Spaces for a live demo

## Author

Built by Varun Golusupudi while learning FastAI and AI engineering.
