from selenium import webdriver
import pandas as pd
import os

class EuDisinfo:
    BASE_URL = "https://euvsdisinfo.eu/"
    DB_PATH = 'disinformation-cases/'
    CASES_PER_PAGE = 100
    COLUMNS = ['date', 'description', 'outlets', 'countries','report_link', 
                'publication_date','languages','countries_discussed', 'keywords', 'links']
    PROXIES = []

    def __init__(self, driver_path):
        self.__driver_path = driver_path

    def get_downloaded_dataset(self, external_data_path):
        claim_reviews = pd.read_csv(os.path.join(external_data_path, "all_claim_reviews.csv"))
        all_countries = pd.read_csv(os.path.join(external_data_path, "all_countries.csv"))
        all_keywords = pd.read_csv(os.path.join(external_data_path, 'all_keywords.csv'))
        all_news_articles = pd.read_csv(os.path.join(external_data_path, "all_news_articles.csv"))
        all_organizations = pd.read_csv(os.path.join(external_data_path, "all_organisations.csv"))
        all_claims = pd.read_json(os.path.join(external_data_path, "claims.json"))

        all_data = []
        for i in range(len(all_claims)):
            data = {}
            item = all_claims.iloc[i]

            # date
            data["date"] = item.datePublished

            #claim data
            claim = item.claimReview
            data["claim"] = claim_reviews.loc[claim_reviews["@id"] == claim].name.values[0]

            #article
            app = item.appearances
            data["links"] = all_news_articles.loc[all_news_articles["@id"].isin(app)].url.to_list()
            authors = all_news_articles.loc[all_news_articles["@id"].isin(app)].author.to_list()
            data["organizations"] = all_organizations.loc[all_organizations["@id"].isin(authors)].name.to_list()

            #keywords
            keywords = item.keywords
            data["keywords"] = all_keywords.loc[all_keywords["@id"].isin(keywords)].name.to_list()

            #countries
            countries = item.contentLocations
            data["countries"] = all_countries.loc[all_countries["@id"].isin(countries)].name.to_list()


            all_data.append(data)
    
        return pd.DataFrame(all_data)
    
    def scrape_database(self, since, until, query):
        #start web driver
        self.__start_driver()

        current_query = self.__build_query(since, until, query)
        self.__get_or_rotate_proxy(current_query)

        data = []
        for i in range(self.CASES_PER_PAGE):
            posts_on_page = self.__driver.find_elements_by_class_name("disinfo-db-post")
            post = posts_on_page[i]
            try: 
                post = self.__process_post(post, current_query)
                print(post)
                data.append(post)
            except Exception as e:
                print(e)
                break
        
        #quit webdriver
        self.__driver.quit()

        return pd.DataFrame(data, columns=self.COLUMNS)

    
    def __process_post(self, post, current_query):
        data = []
        cells = post.find_elements_by_tag_name("td")

        # data
        for i in range(4):
            try:
                data.append(cells[i].text)
            except Exception as e:
                continue

        #get more info on next page
        report_link = cells[1].find_element_by_css_selector('a').get_attribute('href')
        data.append(report_link)

        self.__get_or_rotate_proxy(report_link)
        cols = self.__driver.find_element_by_class_name("b-catalog__sticky").find_elements_by_tag_name("li")

        # data
        for i in range(1,5):
            try:
                data.append(cols[i].text.split("\n")[1])
            except Exception as e:
                continue

        link_list = []
        links = self.__driver.find_elements_by_class_name("b-catalog__link")
        for link in links:
            link_list.append(link.find_element_by_css_selector('a').get_attribute("href"))
        
        #data
        links = ",".join(link_list)
        data.append(links)

        #return to the original page
        self.__get_or_rotate_proxy(current_query)
        return data
    
    def __get_or_rotate_proxy(self, query):
        self.__driver.get(query)

        title = self.__driver.find_element_by_tag_name("title").get_attribute("text")
        print(title)
        if "Access denied" in title or "Attention" in title:
            raise ValueError("scraper was blocked")

    
    def __start_driver(self, proxy=None):
        if proxy is None:
            self.__driver = webdriver.Firefox(executable_path=self.__driver_path)
        else:
            print(proxy)
            options = webdriver.FirefoxOptions()
            options.add_argument(f'--proxy-server={proxy}')
            self.__driver = webdriver.Firefox(options=options, executable_path=self.__driver_path)

    def __build_query(self, since, until, query):
        since = self.__convert_date(since)
        until = self.__convert_date(until)
        return self.BASE_URL+self.DB_PATH+"?text=" + query + "&date=" + since + "%20-%20" + until + "&per_page=" + str(self.CASES_PER_PAGE)

    # converts date to European date for query
    def __convert_date(self, date):
        date = date.split("-")
        date = date[1] + "." + date[0] + "." + date[2]
        return date
