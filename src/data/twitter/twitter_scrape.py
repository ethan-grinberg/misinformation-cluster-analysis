import snscrape.modules.twitter as sntwitter
import pandas

class TwitterScraper:
    def __init__(self, article_url):
        self.__article_url = article_url

    def find_article_tweets(self, limit):
        query = "url:" + self.__article_url
        tweets = []
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i>limit:
                break
            tweets.append(tweet)
        return tweets