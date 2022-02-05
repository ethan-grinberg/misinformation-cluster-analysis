import snscrape.modules.twitter as sntwitter
import pandas as pd

class TwitterScraper:
    def __init__(self, article_url):
        self.__article_url = article_url

    def find_article_tweets(self, limit, since=None):
        query = "url:" + self.__article_url
        if not since is None:
            query = query + " since:" +  since

        return self.__get_tweets_query(query, limit)

    def __get_tweets_query(self, query, limit):
        tweets = []
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i>limit:
                break

            tweet_info = self.__get_tweet_info(tweet)
            user_info = self.__get_user_info(tweet)
            tweets.append({**tweet_info, **user_info})
        
        return pd.DataFrame(tweets)

    # returns a list containing each attribute of the tweet
    def __get_tweet_info(self, tweet_object):
        return  {key:val for key, val in tweet_object.__dict__.items() 
                if not "user" == key and not "media" == key}

    def __get_user_info(self, tweet_object):
        user_dict = tweet_object.user.__dict__
        user_dict["user_id"] = user_dict.pop("id")
        return user_dict
