## Outline
Possible Methodology: Highlight the “possible methods” you could use to solve the problem. Specify how you will be handling/processing the data to train your deep learning pipeline using a CNN model. Furthermore, discuss the metrics that will be used to assess and evaluate the pipeline, and your expectations regarding the kind of results/performance to be achieved. You need to mention the novelty you want to create in the proposal (no need to provide the exact novelty at this point but mentioning some potential direction should be enough)

## Methodology

- **Pipeline**: We plan to make use of the Google Colab platform to train our classifiers. This would ensure that our models are trained faster and allow us to try a larger number of experiments. In order to seamlessly collaborate and transfer changes from our local system to the training system on Colab We will make use of git and Google Drive. All our training data and a copy of our repository will be stored on Drive which will be connected to colab. Using Python, we will read our images, preprocess them and then feed it to the model for training and inference.

- **Pre-processing data**: Different X-Ray scanners have different radiographic contrast. High radiographic contrast leads to density differences that are notably distinguished (black and white). Low radiographic contrast leads to a low-density difference which is difficult to distinguish (black and grey). If our datasets contains scans taken from different machines, there could be a variation in radiographic contrast. We plan to make use of open-source libraries like OpenCV and Pillow to pre-process our images and make it easier for the CNN model to learn.

- **Modeling**: 

- **Optimization**: 

- **Explainability**: 

- **Inference**: 

**Novelty**: As our training dataset is small, we will rely on transfer learning to get a good accuracy on this task. We will be training and/or fine-tuning our best performing model with all the datasets we choose. This would ensure that our model generalizes well and thus can be used as a baseline for all lung related diseases. Additionally, we will try to use a model trained on adult chest X-Rays to predict pneumonia in pediatric chest X-Rays by fine-tuning the weights of the last few layers. 

<br></br>
_________
<br></br>

# References
- https://radiopaedia.org/articles/radiographic-contrast