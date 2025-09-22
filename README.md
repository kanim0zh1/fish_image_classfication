# fish_image_classfication
A deep learning project to classify different types of fish from images using Convolutional Neural Networks (CNNs).This repository includes data preprocessing, model training (VGG16, ResNet, and custom CNN), evaluation using metrics like accuracy, precision, recall, and F1-score, and visualization of model performance.
A Streamlit interface is also provided for easy interaction and real-time predictions.

Key Features:

Data preprocessing and augmentation

Transfer learning with VGG16 and ResNet

Model evaluation with classification reports and confusion matrices

Training history visualization (accuracy & loss)

Streamlit app for real-time fish image classification

Technologies Used:
Python, TensorFlow/Keras, OpenCV, Matplotlib, Seaborn, Streamlit
Models used:

VGG16 (Fine-tuned)

ResNet50 (Fine-tuned)

MobileNet (Fine-tuned)

InceptionV3 (Fine-tuned)

EfficientNetB0 (Fine-tuned)

Custom CNN (from scratch)

Evaluation Metrics:

Accuracy, Precision, Recall, F1-score

Confusion Matrices for each model

📊 Results
Model	Accuracy	Precision	Recall	F1-score
VGG16 (Fine-tuned)	99.21%	99.22%	99.21%	99.06%
ResNet50 (Fine-tuned)	71.82%	76.13%	71.82%	70.31%
MobileNet (Fine-tuned)	99.84%	99.84%	99.84%	99.84%
InceptionV3 (Fine-tuned)	99.81%	99.80%	99.81%	99.79%
EfficientNetB0 (Fine-tuned)	23.00%	24.21%	23.00%	17.15%
Custom CNN (Scratch)	90.49%	90.59%	90.49%	90.36%

✅ Best Model: MobileNet & InceptionV3 (both ~99.8%)

🚀 Deployment (Streamlit App)

Built a Streamlit web app with following features:

Upload fish images (.jpg, .png).

Predict fish category using trained model.

Show confidence score for predictions.

Error handling for non-fish uploads.
