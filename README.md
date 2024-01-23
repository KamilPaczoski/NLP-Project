This project aims to leverage pre-trained language models (LLMs), specifically BERT, for emotion classification in textual data. The system is capable of fine-tuning a pre-trained BERT model on a custom dataset (https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) and using the trained model to predict the emotional tone of input text.

Training: The main program (main.py) allows the fine-tuning of a BERT model on a dataset containing labeled emotional text. The model is trained using PyTorch and the Hugging Face Transformers library.

Text Cleaning: Data preparation involves cleaning the text to improve model performance. The TextCleaner class in TextCleaner.py provides text cleaning functionalities, including removing special characters, lowercase conversion, and other pre-processing steps.

(to do): Inference: After training, the model can be used for emotion classification on new input text. The inference is performed by another program (predictor.py), which loads the trained weights and predicts the emotional tone of a provided text.

Solutions Used
Pre-trained Language Model (LLM): The project utilizes BERT (Bidirectional Encoder Representations from Transformers) as the underlying LLM for understanding contextual relationships in text.

PyTorch: The deep learning framework PyTorch is employed for building, training, and deploying the emotion classification model.

Hugging Face Transformers: This library is used for working with pre-trained language models, including BERT, and simplifies the fine-tuning process.

Text Cleaning: The TextCleaner class in TextCleaner.py is a custom solution for preparing textual data by applying various cleaning operations.
