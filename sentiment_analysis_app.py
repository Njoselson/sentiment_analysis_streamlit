import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import sys
from pathlib import Path
from flair.datasets import ClassificationCorpus
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280


@st.cache
def read_data(data_path, col_names):
    data = pd.read_csv(data_path, header=None, names=col_names,
                       encoding="ISO-8859-1")
    if col_names:
        data = data[['sentiment', 'text']]  # Disregard other columns
    return data


def preprocess(text):
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])[:maxlen]


@st.cache
def prep_data_for_flair(data):
    dat = data.copy()
    dat['text'] = data['text'].apply(preprocess)
    dat['sentiment'] = '__label__' + data['sentiment'].astype(str)
    return dat


# @st.cache
def save_data_for_training(data, amount, data_dir):
    # Create directory for saving data if it does not already exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Save a percentage of the data (you could also only load a fraction of the data instead)
    amount = amount

    data.iloc[0:int(len(data)*0.8*amount)].to_csv(data_dir +
                                                  '/train.csv', sep='\t', index=False, header=False)
    data.iloc[int(len(data)*0.8*amount):int(len(data)*0.9*amount)
              ].to_csv(data_dir + '/test.csv', sep='\t', index=False, header=False)
    data.iloc[int(len(data)*0.9*amount):int(len(data)*1.0*amount)
              ].to_csv(data_dir + '/dev.csv', sep='\t', index=False, header=False)
    return st.write("Data Saved!")


def train_model(data_dir):
    st.write('Creating word corpus for training...')
    corpus = ClassificationCorpus(data_dir)
    label_dict = corpus.make_label_dictionary()
    st.write('Done')

    st.write('Load and create Embeddings for text data...')
    word_embeddings = [WordEmbeddings('glove'),
                       # FlairEmbeddings('news-forward'),
                       # FlairEmbeddings('news-backward')
                       ]
    document_embeddings = DocumentRNNEmbeddings(
        word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
    st.write('Done')

    st.write('Preparing')
    classifier = TextClassifier(
        document_embeddings, label_dictionary=label_dict)
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('model-saves',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=8,
                  max_epochs=200)
    st.write('Model Training Finished!')


def run_the_trainer():
    st.write('Extracting Sentiment Data from Twitter:')
    col_names = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    data_path = 'training.1600000.processed.noemoticon.csv'
    tweet_data = read_data(data_path, col_names)
    st.table(tweet_data.head())

    st.write('Preprocess Data for Flair:')
    tweet_data_processed = prep_data_for_flair(tweet_data)
    st.table(tweet_data_processed.head())

    st.write('Saving processed data for training! How much data should I use?')
    amount = st.slider('Amount', 0.0, 100.0, 12.5)
    data_dir = './processed-data'
    save_data_for_training(tweet_data_processed, amount/100, data_dir)

    final_model = Path('model-saves/final-model.pt')
    if final_model.is_file():
        st.write('You already have a saved model! Do you want to retrain it?')
        if st.button('Rerun'):
            st.write('Train Model!')
            train_model(data_dir)
        else:
            st.write("Don't train model!")
            pass
    else:
        st.write('Train Model!')
        train_model(data_dir)


def run_the_app():
    st.subheader('Single sentence classification')

    tweet_input = st.text_input('Sentence:')
    classifier = TextClassifier.load('model-saves/final-model.pt')
    sentence_data = Sentence(preprocess(tweet_input))

    classifier.predict(sentence_data)
    label_dict = {'0': 'Negative', '4': 'Positive'}
    if len(sentence_data.labels) > 0:
        st.write('Prediction:')
        st.write(label_dict[sentence_data.labels[0].value] + ' with ',
                 sentence_data.labels[0].score*100, '% confidence')


def add_new_data_():

    df1 = pd.DataFrame(
        np.random.randn(50, 20),
        columns=('col %d' % i for i in range(20)))

    my_table = st.table(df1)
    if st.button('Add new Data'):
        df2 = pd.DataFrame(
            np.random.randn(50, 20),
            columns=('col %d' % i for i in range(20)))

        my_table.add_rows(df2)


def add_new_data():
    label_dict = {'Negative': '0', 'Positive': '4'}

    st.subheader('Single sentence classification')

    # We collect inputs with text_input boxes for sentences and their corresponding sentiments
    sentence_data = st.text_input('Sentence:')
    classification_label = st.selectbox('Sentence:', ['Negative', 'Positive'])

    sentiment = label_dict.get(classification_label)
    classification_data = pd.DataFrame(
        data={'sentiment': [sentiment], 'text': [sentence_data]})
    prepped_data = prep_data_for_flair(classification_data)

    st.write('This is the most recent sentiment data we have:')
    data_path = './processed-data/train.csv'
    tweet_data = pd.read_csv(data_path, None)
    last_data_table = st.table(tweet_data.tail())
    if st.button('Add new Data'):
        st.write('New data added to the training set!')
        last_data_table.add_rows(prepped_data)
        prepped_data.to_csv('./processed-data/train.csv',
                            sep='\t',  mode='a', index=False, header=False)
        st.balloons()


def main():
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Train the Model", "Run the app", "Add some new Data"])
    if app_mode == "Show instructions":
        st.sidebar.success(
            'To continue select "Train the Model", "Run the app" or "Add some new Data"')
    elif app_mode == "Train the Model":
        run_the_trainer()
    elif app_mode == "Run the app":
        run_the_app()
    elif app_mode == "Add some new Data":
        add_new_data()


if __name__ == "__main__":
    main()
