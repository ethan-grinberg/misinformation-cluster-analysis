from selenium import webdriver

class ScrapeEU:
    BASE_URL = "https://euvsdisinfo.eu/disinformation-cases/"
    CASES_PER_PAGE = 100

    def __init__(self, driver_path):
        self.__driver_path = driver_path

    def build_query(self, since, until, query):
        since = self.__convert_date(since)
        until = self.__convert_date(until)
        return self.BASE_URL+"?text=" + query + "&date=" + since + "%20-%20" + until + "&per_page=" + str(self.CASES_PER_PAGE)

    # converts date to European date for query
    def __convert_date(self, date):
        date = date.split("-")
        date = date[1] + "." + date[0] + "." + date[2]
        return date


