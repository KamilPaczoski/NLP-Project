This project aims to leverage pre-trained language models (LLMs), specifically BERT, for emotion classification in textual data. The system is capable of fine-tuning a pre-trained BERT model on a custom dataset (https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) and using the trained model to predict the emotional tone of input text.

Training: The main program (initialize.py) allows the fine-tuning of a BERT model on a dataset containing labeled emotional text. The model is trained using PyTorch and the Hugging Face Transformers library.

Text Cleaning: Data preparation involves cleaning the text to improve model performance. The TextCleaner class in TextCleaner.py provides text cleaning functionalities, including removing special characters, lowercase conversion, and other pre-processing steps.

Inference: After training, the model can be used for emotion classification on new input text. The inference is performed by another program (emotion_evaluator.py), which loads the trained weights and predicts the emotional tone of a provided text.

Solutions Used

Pre-trained Language Model (LLM): The project utilizes BERT (Bidirectional Encoder Representations from Transformers) as the underlying LLM for understanding contextual relationships in text.

PyTorch: The deep learning framework PyTorch is employed for building, training, and deploying the emotion classification model.

Hugging Face Transformers: This library is used for working with pre-trained language models, including BERT, and simplifies the fine-tuning process.

Text Cleaning: The TextCleaner class in text_cleaner.py is a custom solution for preparing textual data by applying various cleaning operations it mostly utilizes solutions provided by nltk library


In order to run the project, the following steps are required:
if you have not pregenerated weights for the model (saved_weights.pt), you can run initialize.py to train the model
run flask_app.py to start the web interface for the model
as for now the user can use the web interface to input comments and get the predicted emotion and even update model with provided feedback 
fine-tuning using feedback feature as for now is not advised as singular inputs are not enough to provide significant improvement to the model in worst case scenario it can even decrease the model's performance

## to do list
implement recurrent data cleaning (with text_cleaner) of feedback data as new entries are added to the database via web interface
model has slight problems with surprise and love it is still sometimes wrongly classified as joy