import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
    return text


TRAIN_CSV_PATH = "./files/train_short.csv"
TEST_CSV_PATH = "./files/test.csv"

df = pd.read_csv(TRAIN_CSV_PATH)

# get categories
categories = df.category.unique()
# split by Label quality
df_ok = df[df['label_quality'] == 'reliable']
df_doubt = df[df['label_quality'] == 'unreliable']

print(categories)
print("Categories: " + str(categories.size))
print("Total data: " + str(df.size))
print("Data with Label quality verified: " + str(df_ok.size))
print("Data with Label quality not verified: " + str(df_doubt.size))

# clean text
df['title'] = df['title'].apply(clean_text)

train_size = int(df.size * .7)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (df.size - train_size))

train_title = df['title'][0:train_size]
train_category = df['category'][0:train_size]

test_title = df['title'][train_size:]
test_category = df['category'][train_size:]

print('train_title shape:', train_title.shape)
print('train_category shape:', train_category.shape)
print('test_title shape:', test_title.shape)
print('test_category shape:', test_category.shape)

max_words = 2000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_title) # only fit on train
x_train = tokenize.texts_to_matrix(train_title)
x_test = tokenize.texts_to_matrix(test_title)

encoder = LabelEncoder()
encoder.fit(train_category)
y_train = encoder.transform(train_category)
y_test = encoder.transform(test_category)


num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

batch_size = 32
epochs = 2

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print(score)
# print('Test accuracy:', score[1])