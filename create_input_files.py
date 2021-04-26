from utils import create_input_files, train_word2vec_model

if __name__ == '__main__':

    """
    create_input_files(csv_folder='./.data',
                       output_folder='./.data',
                       sentence_limit=15,
                       word_limit=20,
                       min_word_count=5)
    """

    train_word2vec_model(data_folder='./.data', algorithm='skipgram')
