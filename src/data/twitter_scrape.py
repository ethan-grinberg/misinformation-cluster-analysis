import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np

class TwitterScraper:
    def __init__(self):
        pass

    def find_tweets_all_articles(self, claim_df, limit=None, per_tweet_limit=500):
        until = limit
        if limit is None:
            until = len(claim_df)
        
        dfs = []
        for i in range(until):
            links = claim_df.iloc[i].links
            for link in links:
                try:
                    tweets = self.find_article_tweets(link, per_tweet_limit)
                    tweets["claim"] = claim_df.iloc[i].claim
                    dfs.append(tweets)
                except Exception as e:
                    print(e)
                    continue
        
        all_tweets = pd.concat(dfs)
        return all_tweets

    def find_article_tweets(self, q, limit, since=None, article=True):
        query = None
        if article:
            query = "url:" + q
        else:
            query = q
            
        if not since is None:
            query = query + " since:" +  since

        tweets =  self.__get_tweets_query(query, limit)
        if not len(tweets) == 0:
            self.__clean_df(tweets)
        return tweets

    def __clean_df(self, df):
        df["id"] = df["id"].astype(np.int64)
        df["user_id"] = df["user_id"].astype(np.int64)

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
