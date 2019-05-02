from textblob import TextBlob
import nltk

nltk.download('popular')

wiki = TextBlob("Siraj is angry that he never gets good matches on Tinder")
print(wiki.tags)
print(wiki.words)
print(wiki.sentiment.polarity)


