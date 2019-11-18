import pandas as pd
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def tokenize(df, column_name, token_column_name):
    """
    Creating a tokenized, stemmed, removed stopped words version of given column name

    :param df: dataframe
    :param column_name: Text column to tokenize
    :param token_column_name: Output column which contains tokens
    :return: dataframe with tokenized columns
    """

    # Lowercase
    df[token_column_name] = df[column_name].str.lower()

    # Tokenize
    df[token_column_name] = df[token_column_name].apply(word_tokenize)

    # Stemming
    porter = PorterStemmer()
    df[token_column_name] = df[token_column_name].apply(lambda x: list(map(porter.stem, x)))

    # Removing Stop Words
    stop_words = set(stopwords.words('english'))
    df[token_column_name] = df[token_column_name].apply(lambda x: [w for w in x if w not in stop_words])

    return df


def remove_words_from_column(df, column_name, to_remove):
    """
    Remove a series of words, typically used to remove common words that normally don't carry any meaning.

    :param df: input dataframe
    :param column_name: column to apply changes to
    :param to_remove: series of words to remove
    :return: dataframe
    """
    df[column_name] = df[column_name].apply(lambda x: [w for w in x if w not in to_remove])
    return df


def tokens_to_vec(df, column_name, max_length):
    """
    Take a column of tokens and convert them to a list of tokens based on the pre-trained tokenizer built from the
    training data.  It also pads all resulting vector lists to meet the same max length specified.

    :param df: input dataframe
    :param column_name: target column containing list of tokens
    :param max_length: the maximum length of tokens allowed into the neural net
    :return: the list of vectors
    """
    truncated_lists = list(df[column_name].str[:max_length])

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    vecs_list = tokenizer.texts_to_sequences(truncated_lists)

    # Sanitize words that might not be found in the training set
    vecs_list = [vecs for vecs in vecs_list if len(vecs) > 0]

    return pad_sequences(vecs_list, maxlen=max_length, padding='post')


def predict_from_model(x):
    """
    Using the trained neural net to predict the results.

    :param x: List of vectors
    :return: probability matrix of each result
    """
    model = load_model('./keras_try1.h5')
    return model.predict(x)


def add_prediction_to_df(df, prediction_matix, prediction_column_name, keep_columns):
    """
    Final cleanup stage and adding prediction to output dataframe.  The input dataframe is cleaned up with the keep
    columns specified. Then the max predictions are used from the prediction matrix, and attached to the dataframe with
    the given prediction column name.

    :param df: input dataframe
    :param prediction_matix: 2d array of all the rows and probabilities for each event
    :param prediction_column_name: the column name for the prediction
    :param keep_columns: the columns that will be kept in the output model (with the exception of the prediction column)
    :return: output dataframe
    """
    prediction = prediction_matix.argmax(1)
    decoded_predictions = decode_prediction(prediction)
    final_df = df[keep_columns]
    final_df[prediction_column_name] = decoded_predictions
    return final_df


def decode_prediction(predictions):
    with open('encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    return encoder.inverse_transform(predictions)


def write_output(df):
    """
    Writing the dataframe to csv for the final submission format.  This tries to write the `/solution/` directory,
    as it serves as the volume mounted in the docker image.  If that directory doesn't exist, it just writes to
    the current directory.

    :param df: input dataframe
    :return:
    """
    output_dir = "/solution/" if os.path.exists("/solution") else ""
    print(f"output_dir: {output_dir}")

    df.to_csv(f'{output_dir}solution.csv', index=False)


def write_output2(df):
    import csv

    with open('../solution/solution2.csv', 'w') as csv_output:
        writer = csv.DictWriter(csv_output, ['text', 'sex', 'age', 'event'])
        writer.writeheader()
        for index, row in df.iterrows():
            csv_row = {'text': row['text'], 'sex': row['sex'], 'age': row['age'], 'event': row['event']}
            writer.writerow(csv_row)


if __name__ == '__main__':
    # Read test data
    df_test = pd.read_csv('test.csv')

    # Save the original columns as they will be needed for the final output
    original_columns = list(df_test.columns)

    # Tokenize 'text' column
    df_tokens = tokenize(df_test, 'text', 'text_edit')

    # Clean up words without significant meaning from token fields
    cleaned_df = remove_words_from_column(df_tokens, 'text_edit',
                                          to_remove={'at', 'a', 'onto', 'state', 'yf', 'm', 'f', 'to', 'on', 'male',
                                                     'female', 'yom', 'yof', 'yom', 'pt', 'yo', 'ym', 'rt', 'lt', 'o',
                                                     'l', 'r', 'left', 'right', })
    # Convert tokens to vectors
    x_test = tokens_to_vec(cleaned_df, 'text_edit', 15)

    # Run vectors through neural net to predict events
    prediction_output = predict_from_model(x_test)

    # Clean up prediction and dataframe for output csv
    prediction_df = add_prediction_to_df(cleaned_df, prediction_output, "event", original_columns)

    # Write output csv
    write_output(prediction_df)
