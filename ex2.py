# Siraj Raval - Learn Python for Data Science 2
import tweepy
from textblob import TextBlob

consumer_key = 'FHXdD0vLTJwqlCsMnKKJ5Fq4W'
consumer_secret = 'lmnXutLWC9fiB1Nt0GycELfnd0SjNBIPkLnU6N7pc7OfTK1rKn'

access_token = '14133556-Av3JD8lpGWn4uUncIYMFlusswa9WreFwsWDg70pA5'
access_token_secret = 's7Oh8CpvDRi5P93cQNw02xJyiWzKmcf8OGVLYtSVVs7FF'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

# create a labeled tweet dataset csv format