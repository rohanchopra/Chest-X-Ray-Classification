## Due Date, Next Meeting - Thursday, October 27, 2022

### Data Pre-processing
1. **Jolly** - Pneumonia, COVID-19 dataset
2. **Abhishek** - Pneumonia dataset
3. **Rohan**- Chest X-Ray 8 Dataset

All pre-processing needs to be done using PyTorch
Pre-processing functions to write
- Re-size image
- Flip image horizontally
- Change contrast of the image
- Histogram equalization + gaussian blur (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949)
- Adaptive masking (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949)
- Below are more optional pre-processing which can be done for better results (Source Kaggle https://www.kaggle.com/code/soumya9977/deep-dive-in-image-preprocessing-using-opencv)
- Basic Geometric Transformations
  - Image Translation
  - Image Rotation
  - Image Cropping
- Color Space Correction
  - RGB Color Space
  - HSV Color Space
  - HSL Color Space
- Blurring and Smoothing
  - Gaussian Blurring
  - Median Blurring
- Morphology (Smoothing edges https://prince-canuma.medium.com/image-pre-processing-c1aec0be3edf)
- Remove Noise (Denoise)

### Model Explainability
**Harman**

- Pick a working notebook from the #1 Pneumonia, COVID-19 dataset (consult Jolly) and run it end-to-end to get the trained model and predictions.
- Apply SHAP on this model and generate visualizations
- Apply GradCAM on this model and generate visualizations
