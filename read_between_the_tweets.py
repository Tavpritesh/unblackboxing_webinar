import pandas as pd

from sklearn.externals import joblib

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from twitter_sentiment.preprocessing import tweet_train_test_split, TweetPreprocessor
from twitter_sentiment.model import TweetClassifier, TweetClassifierNeptune
from twitter_sentiment.architectures import arch_lstm, arch_conv1d, arch_attention, arch_attention36


NEPTUNE = True
MODEL_FILEPATH = '/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/\
models/glove.twitter.27B.25d.txt'
DATA_FILEPATH = '/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/\
data/tweets/tweet_sentiment_dataset.csv'
PREP_DUMP_FILEPATH = '/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/\
models/tweet_preprocessor.pkl'
CLASS_DUMP_FILEPATH ='/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/\
models/tweetnetAttention36.h5py'
MAX_WORDS = 20000
MAX_SEQ_LEN = 30
EMBEDDING_DIM = 25
ARCHITECTURE = arch_attention36


if __name__ == '__main__':
    
    tweet_dataset = pd.read_csv(DATA_FILEPATH)

    (X_train, y_train), (X_test,y_test) = tweet_train_test_split(tweet_dataset, train_size=0.8,
                                                                random_state=1234)
    
    tweet_prep = TweetPreprocessor(max_nr_words=MAX_WORDS, max_sequence_length=MAX_SEQ_LEN)
    X_train, y_train = tweet_prep.fit_transform(X=X_train['tweet'].values, y=y_train)
    X_test, y_test = tweet_prep.transform(X=X_test['tweet'].values, y=y_test)
    joblib.dump(tweet_prep, PREP_DUMP_FILEPATH)
    
    if NEPTUNE:
        tweet_classifier = TweetClassifierNeptune(architecture=ARCHITECTURE,
                                                  max_nr_words=MAX_WORDS,
                                                   sequence_length=MAX_SEQ_LEN,
                                                   embedding_dim=EMBEDDING_DIM,
                                                   path_to_word_embeddings=MODEL_FILEPATH,
                                                   word_index = tweet_prep.tokenizer.word_index,
                                                   classes=2,
                                                   model_save_filepath=CLASS_DUMP_FILEPATH)
    else:
        tweet_classifier = TweetClassifier(architecture=ARCHITECTURE,
                                           max_nr_words=MAX_WORDS,
                                           sequence_length=MAX_SEQ_LEN,
                                           embedding_dim=EMBEDDING_DIM,
                                           path_to_word_embeddings=MODEL_FILEPATH,
                                           word_index = tweet_prep.tokenizer.word_index,
                                           classes=2,
                                           model_save_filepath=CLASS_DUMP_FILEPATH)       
    
    tweet_classifier.train((X_train, y_train), (X_test, y_test), batch_size=128, epochs=10, verbose=2)