import torch
from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

with open('train.txt', 'r') as file:
    data = file.readlines()
data = [line.strip().split(';') for line in data]
df_train = pd.DataFrame(data, columns=['sentence', 'label'])


class TextCleaner():
    def __init__(self, df_input):
        self.cleaned_text = self.for_df(df_input)

    def remove_numbers(self, text):
        text = ''.join([i for i in text if not i.isdigit()])
        return text

    def remove_punctuation(self, text):
        text = ''.join([i for i in text if i not in frozenset(string.punctuation)])
        return text

    def remove_whitespace(self, text):
        text = ' '.join(text.split())
        return text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        text = ' '.join(
            [i for i in text.split() if i.lower() not in stop_words])
        return text

    def lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    def text_cleaning(self, text):
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.remove_whitespace(text)
        text = self.remove_stopwords(text)
        text = self.lemmatization(text)
        return text

    def for_df(self, df):
        df['sentence'] = df['sentence'].apply(self.text_cleaning)
        return df


if __name__ == '__main__':
    print(df_train.head())
    print(TextCleaner(df_train).cleaned_text.head())
