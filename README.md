In order to run the project, the following steps are required:

if you have not pregenerated weights for the model (saved_weights.pt), you can run initialize.py to train the model

run flask_app.py to start the web interface for the model

as for now the user can use the web interface to input comments and get the predicted emotion and even update model with provided feedback 

fine-tuning using feedback feature as for now is not advised as singular inputs are not enough to provide significant improvement to the model in worst case scenario it can even decrease the model's performance

## to do list
implement recurrent data cleaning (with text_cleaner) of feedback data as new entries are added to the database via web interface
model has slight problems with surprise and love it is still sometimes wrongly classified as joy
