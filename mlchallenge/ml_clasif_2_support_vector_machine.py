import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

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


TRAIN_CSV_PATH1 = "./files/train_esp_ok.csv"
TRAIN_CSV_PATH2 = "./files/train_por_ok.csv"
TRAIN_CSV_PATH3 = "./files/train_esp_doubt.csv"
TRAIN_CSV_PATH4 = "./files/train_por_doubt.csv"

TRAIN_SVM_CLASSIF1 = "./files/svm_clasif_esp_ok.pkl"
TRAIN_SVM_CLASSIF2 = "./files/svm_clasif_por_ok.pkl"
TRAIN_SVM_CLASSIF3 = "./files/svm_clasif_esp_doubt.pkl"
TRAIN_SVM_CLASSIF4 = "./files/svm_clasif_por_doubt.pkl"


def train_clasif(csv_file, svm_clasif_file):
    df = pd.read_csv(csv_file)

    # get categories
    categories = df.category.unique()

    print("Categories: " + str(categories.size))
    print("Total data: " + str(df.size))

    # clean text
    df['title'] = df['title'].apply(clean_text)

    X = df.title
    y = df.category
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                   ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred))
    # accuracy 0.845103511023296

    # save the classifier
    with open(svm_clasif_file, 'wb') as fid:
        pickle.dump(sgd, fid)


# start program
# run clasif 1
train_clasif(TRAIN_CSV_PATH1, TRAIN_SVM_CLASSIF1)
# run clasif 2
train_clasif(TRAIN_CSV_PATH2, TRAIN_SVM_CLASSIF2)
# run clasif 3
train_clasif(TRAIN_CSV_PATH3, TRAIN_SVM_CLASSIF3)
# run clasif 4
train_clasif(TRAIN_CSV_PATH4, TRAIN_SVM_CLASSIF4)


