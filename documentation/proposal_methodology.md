## Outline
Possible Methodology: Highlight the “possible methods” you could use to solve the problem. Specify how you will be handling/processing the data to train your deep learning pipeline using a CNN model. Furthermore, discuss the metrics that will be used to assess and evaluate the pipeline, and your expectations regarding the kind of results/performance to be achieved. You need to mention the novelty you want to create in the proposal (no need to provide the exact novelty at this point but mentioning some potential direction should be enough)


## Proposal
### Methodology

We plan to make use of the Google Colab platform to train our classifiers. In order to seamlessly collaborate and transfer changes from our local system to the training system on Colab We will make use of git and Google Drive. 

**Pre-processing**: Different X-Ray scanners have different radiographic contrast. As our datasets are from different sources, we plan on using libraries like OpenCV and Pillow to pre-process our images and make it easier for our CNN models to learn.

**Model and Training**: We plan on using the following models:
- **VGG 16**: A classical deep CNN architecture with a high number of trainable parameters. High number of layers might help our model pick up specific details in X-Ray scans.
- **Inception V3**: This model uses multiple filters of different sizes on the same level to prevent overfitting, which is highly probable as the size of our dataset is small. This model has a low number of trainable parameters which will help us experiment more and find the perfect hyperparameters.
- **Custom CNN model**: After looking at results from the previous two models, we plan on creating our own architecture that we expect to perform better, either on training time or on the accuracy front.

To train our models, we will use cross-entropy loss function and the Adam optimizer, for which we will experiment with manual learning rate decay. As the size of our datasets is small, we plan on experimenting with transfer learning to reduce our training time and improve results.

**Evaluation**: Given that our datasets are highly imbalanced, we will not rely on accuracy and will primarily use the confusion matrix. Additionally, we plan on calculating the weighted F-measure with a //beta// of 2 (higher precision) for each class of diseased lung compared with a healthy lung. 

To find the best hyperparameters, we will perform ablation studies and also make use of Ray Tune. To explain the results of our models, we plan on using SHAP and GradCAM which will help us diagnose our models and help end users get more confidence on our model's decisions. 

We look forward to having a model that can predict diseases in lung X-Rays with an F2-measure above 0.8.



## Long-Form
### Methodology

- **Pipeline**: We plan to make use of the Google Colab platform to train our classifiers. This would ensure that our models are trained faster and allow us to try a larger number of experiments. In order to seamlessly collaborate and transfer changes from our local system to the training system on Colab We will make use of git and Google Drive. All our training data and a copy of our repository will be stored on Drive which will be connected to colab. Using Python, we will read our images, preprocess them and then feed it to the model for training and inference.

- **Pre-processing data**: Different X-Ray scanners have different radiographic contrast. High radiographic contrast leads to density differences that are notably distinguished (black and white). Low radiographic contrast leads to a low-density difference which is difficult to distinguish (black and grey). If our datasets contains scans taken from different machines, there could be a variation in radiographic contrast. We plan to make use of open-source libraries like OpenCV and Pillow to pre-process our images and make it easier for the CNN model to learn.

- **Modeling**: We plan on using the following models
     - **VGG 16**: A classical deep CNN architecture with a high number of trainable parameters. High number of layers might help our model pick up specific details in X-Ray scans.
    - **Inception V3**: This model uses multiple filters of different sizes on the same level to prevent overfitting, which is highly probable as the size of our dataset is small. This model has a low number of trainable parameters which will help us experiment more and find the perfect hyperparameters.
    - **Custom CNN model**: After looking at results from the previous two models, we plan on creating our own architecture that we expect to perform better, either on training time or on the accuracy front.

    As the size of our datasets is small, we plan on experimenting with transfer learning so as to reduce our training time and improve results.
    
    We plan on using cross-entropy loss function and the Adam optimizer with a learning rate decay due to its fast computation. 

- **Optimization**: 

- **Evaluation**: Given that our datasets are highly imbalanced, we will not rely on accuracy and will primarily use the confusion matrix. Additionally, we plan on calculating the weighted F-measure with a //beta// of 2 (higher precision) for each class of diseased lung compared with a healthy lung. 

- **Explainability**: To explain the results of our models, we plan on using SHAP and GradCAM which will help us diagnose our models and help end users get more confidence on our model's decisions. 


**Novelty**: As our training dataset is small, we will rely on transfer learning to get a good accuracy on this task. We will be training and/or fine-tuning our best performing model with all the datasets we choose. This would ensure that our model generalizes well and thus can be used as a baseline for all lung related diseases.



<br></br>
_________
<br></br>

# References
- https://radiopaedia.org/articles/radiographic-contrast
- https://pytorch.org/vision/stable/models.html#classification
- https://paperswithcode.com/lib/torchvision/vgg
- https://paperswithcode.com/lib/torchvision/inception-v3
- https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
- https://github.com/jacobgil/pytorch-grad-cam
- https://github.com/slundberg/shap
- https://machinelearningmastery.com/fbeta-measure-for-machine-learning/
- https://docs.ray.io/en/latest/tune/index.html
