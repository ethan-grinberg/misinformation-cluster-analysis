from selenium import webdriver
import pandas as pd

class ScrapeEU:
    BASE_URL = "https://euvsdisinfo.eu/"
    DB_PATH = 'disinformation-cases/'
    CASES_PER_PAGE = 100
    COLUMNS = ['date', 'description', 'outlets', 'countries','report_link', 
                'publication_date','languages','countries_discussed', 'keywords', 'links']
    PROXIES = ['70.60.132.34:1080',
                '70.169.129.246:48678',
                '97.90.49.141:54321',
                '74.214.177.61:8080',
                '64.227.2.136:8080',
                '54.174.159.255:8080',
                '68.183.237.129:8080',
                '70.169.70.90:80',
                '96.66.200.209:64312',
                '54.213.173.231:80']

    def __init__(self, driver_path):
        self.__driver_path = driver_path
        webdriver.FirefoxOptions()
    
    def search_database(self, since, until, query, limit):
        #start web driver
        self.__start_driver()

        current_query = self.__build_query(since, until, query)
        self.__get_or_rotate_proxy(current_query)

        data = []
        for i in range(self.CASES_PER_PAGE):
            posts_on_page = self.__driver.find_elements_by_class_name("disinfo-db-post")
            post = posts_on_page[i]
            data.append(self.__process_post(post, current_query))
        
        #quit webdriver
        self.__driver.quit()

        return pd.DataFrame(data, columns=self.COLUMNS)

    
    def __process_post(self, post, current_query):
        data = []
        cells = post.find_elements_by_tag_name("td")

        # data
        NUM_CELL_DATA = 3
        for i in range(NUM_CELL_DATA):
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
        for i in range(len(self.PROXIES) - 1):
            print(i)
            self.__driver.get(query)

            title = self.__driver.find_element_by_tag_name("title").get_attribute("text")
            if "Access denied" in title:
                if i == (len(self.PROXIES) - 1):
                    raise ValueError("all proxies blocked")
                else:
                    self.__driver.quit()
                    self.__start_driver(proxy=self.PROXIES[i])
            else:
                break

    
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


