import tweepy
import json
import pandas as pd

# Twitter API credentials
api_key = "8i8RsaEZKD7XQGMFWscSu07li"
api_secret_key = "zogU75qnVGVHVlQ8Ke074V0uvg6Yh9pp4N8Rxwx2vfKBHyHl5S"
access_token = "1550200058477481987-SApDlK998C0mYyaq2knLxXmJ3saxQ2"
access_token_secret = "5B4NUOpuyxZO8HXszzEbohdsC0jcAE8e2Aoflz7TX70vx"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Define the search query
search_query = "sosyal anksiyetem -filter:retweets"

# Collect tweets
tweets = tweepy.Cursor(api.search_tweets,
                       q=search_query,
                       lang="tr",
                       tweet_mode='extended').items(1000)

# Create a list to store

# Create a list to store tweet texts
tweets_list = []

for tweet in tweets:
    tweets_list.append(tweet.full_text)

# Convert list to DataFrame
df = pd.DataFrame(tweets_list, columns=['post_body'])

# Display the DataFrame
print(df.head())

# Save DataFrame to a CSV file
df.to_csv("scrapped_social_anxiety_tweets.csv", index=False)