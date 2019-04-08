import tweepy
import numpy as np
import pandas as pd
import jsonlines

consumer_key = "" 
consumer_secret = ""
access_key = ""
access_secret = ""


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
tweets = tweepy.Cursor(api.user_timeline, screen_name='midasIIITD')
#tweets = api.user_timeline(screen_name='midasIIITD')
json = []

for tweet in tweets.items():
##    print(tweet._json)
    json.append(tweet._json)

with jsonlines.open('Saaransh_Pandey.jsonl', 'w') as writer:
    writer.write_all(json)


tweets_text = []
tweets_created_at = []
tweets_favorites_count = []
tweets_retweets_count = []
tweets_images_count = []
##is_retweeted = []
##original_created_at = []
##original_retweets_count = []
##original_favorites_count = []
for tweet in json:
    images_count = 0
    tweets_text.append(tweet['text'])
    tweets_created_at.append(tweet['created_at'])
    tweets_favorites_count.append(tweet['favorite_count'])
    tweets_retweets_count.append(tweet['retweet_count'])

##    if 'retweeted_status' in tweet:
##        is_retweeted.append('Yes')
##        retweeted_status = tweet['retweeted_status']
##        original_created_at.append(retweeted_status['created_at'])
##        original_retweets_count.append(retweeted_status['retweet_count'])
##        original_favorites_count.append(retweeted_status['favorite_count'])
##        
##    else:
##        is_retweeted.append('No')
##        original_created_at.append(None)
##        original_retweets_count.append(None)
##        original_favorites_count.append(None)

        
    entities = tweet['entities']
##    print(entities)

    for media in entities.get("media",[{}]):
        if media.get('type',None) == 'photo':
            images_count+=1

    if images_count == 0:
        tweets_images_count.append(None)
        print(tweets_images_count)
    else:
        tweets_images_count.append(images_count)
        print(images_count)
            


field_names = ['Text','Date and Time', 'Favorites/Likes Count',
                          'Retweets Count','Images Count']

##field_names = ['Text','Date and Time', 'Favorites/Likes Count',
##                          'Retweets Count','Images Count', 'Is Retweeted?',
##               'Original Date&Time','Original Favorites Count',
##               'Original Retweets Count']

data = {'Text':tweets_text, 'Date and Time': tweets_created_at,
        'Favorites/Likes Count': tweets_favorites_count,
        'Retweets Count':tweets_retweets_count,
        'Images Count':tweets_images_count}

##data = {'Text':tweets_text, 'Date and Time': tweets_created_at,
##        'Favorites/Likes Count': tweets_favorites_count,
##        'Retweets Count':tweets_retweets_count,
##        'Images Count':tweets_images_count,
##        'Is Retweeted?':is_retweeted,
##        'Original Date&Time':original_created_at,
##        'Original Favorites Count':original_favorites_count,
##        'Original Retweets Count':original_retweets_count}


##pd.set_option('display.max_columns', 9)
pd.set_option('display.max_columns', 5)
df = pd.DataFrame(data, columns = field_names)
##print(df)


##print(tweets_text)
##print(tweets_created_at)
##print(tweets_favorites_count)
##print(tweets_retweets_count)
##print(tweets_images_count)
    


##raw_data = {'json': json}
##df = pd.DataFrame(raw_data, columns = ['json'])
##df.index.name = 'index'
##df.to_csv('./Saaransh_Pandey.csv') 

