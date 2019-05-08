# Siraj Raval - Learn Python for Data Science 2
import tweepy
from textblob import TextBlob
from private import PersTwitterKey

consumer_key = PersTwitterKey.TW_CONSUMER_KEY.value
consumer_secret = PersTwitterKey.TW_CONSUMER_SECRET.value

access_token = PersTwitterKey.TW_ACCESS_TOKEN.value
access_token_secret = PersTwitterKey.TW_ACCESS_TOKEN_SECRET.value

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

# create a labeled tweet dataset csv format