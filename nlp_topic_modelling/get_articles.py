import os
import json
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


class Scroll:

    def __init__(self, driver):
        """ javascript scroller class
        :param driver: chrome driver
        """
        self.javascript_scroll = "window.scrollTo(0, document.body.scrollHeight);" \
                                 "var body = document.body, html = document.documentElement;" \
                                 "var height = Math.max(" \
                                 "body.scrollHeight, body.offsetHeight, html.clientHeight, html.scrollHeight, html.offsetHeight" \
                                 ");" \
                                 "return height"
        self.driver = driver

    def detect_end_of_scroll(self):
        """ scroll down to the end of a web page using javascript """
        # print('scrolling to end of page', end='')
        len_of_page = self.execute_scroll()
        last_count = -1
        while last_count != len_of_page:
            last_count = len_of_page
            len_of_page = self.execute_scroll()
        # print('...finished scroll')

    def scroll_x_times(self, x):
        """ scroll down x amount of times
        :param x: number of times to scroll
        """
        # print('scrolling {} times...'.format(x))
        for _ in range(x):
            self.execute_scroll()

    def execute_scroll(self):
        """ execute javascript scroll down method
        :return: current length of page
        """
        len_of_page = self.driver.execute_script(self.javascript_scroll)
        return len_of_page


class TechCrunchArticleScraper:

    def __init__(self):
        self.driver = webdriver.Chrome(options=self.configure_driver_options(headless=True))
        self.scroll = Scroll(self.driver)

    @staticmethod
    def configure_driver_options(headless):
        """ configure chrome driver options
        :param headless: running headless will not open browser
        :return: chrome options
        """
        options = Options()
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        if headless:
            options.add_argument("--headless")
        options.add_experimental_option('prefs', {
            'credentials_enable_service': False,
            'profile': {
                'password_manager_enabled': False
            }
        })
        return options

    def get_soup(self):
        """ get TC homepage beautiful soup
        :return: home page soup
        """
        self.driver.get('https://techcrunch.com/')
        elem = self.driver.find_element_by_name("agree")
        elem.send_keys(Keys.RETURN)
        count = 0
        print('iteration', end=' ')
        while count < 1000:
            self.scroll.scroll_x_times(1)
            elem = self.driver.find_element_by_class_name("load-more")
            elem.send_keys(Keys.RETURN)
            if count % 10 == 0:
                print(f'{count}', end=' ')
            count += 1
        print('\n soup retrieved...')
        # req.add_header('EuConsent', str(uuid.uuid4()))
        # req.add_unredirected_header('EuConsent', str(uuid.uuid4()))
        # gcontext = ssl.SSLContext()
        # res = request.urlopen(req, context=gcontext)
        # res = driver.execute_script('return document.documentElement.outerHTML')
        return BeautifulSoup(self.driver.page_source, 'html.parser')

    def close_driver(self):
        """ quit selenium webdriver - close chrome """
        self.driver.quit()

    @staticmethod
    def get_all_article_links(_soup):
        """ get all links to TC posts
        :param _soup: TC beautiful soup
        :return: dictionary of links
        """
        print('getting lateset Tech Crunch article links...')
        _article_links = {}
        for a in _soup.find_all('a', {'class': 'post-block__title__link'}):
            if a.has_attr('href'):
                _article_links[a.text] = a['href']
        print(f'{len(_article_links)} article links found')
        return _article_links

    def extract_article_content(self, article_dict):
        """ get all text posts from TC
        :param article_dict: dictionary of article titles and urls
        :return: _articles - titles and content
        """
        print('extracting Tech Crunch article content...')
        _articles = {}
        print(f'article', end=' ')
        for i, title in enumerate(article_dict):
            url = article_dict[title]
            if title == '':
                continue
            if i % 10 == 0:
                print(f'{i}', end=' ')
            content = ''
            self.driver.get(f'https://techcrunch.com/{url}')
            article_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            post_body = article_soup.find('div', {'class': 'article-content'})
            try:
                for para in post_body.findAll('p'):
                    content += para.text  # posts += [p.text.replace('?', ' ') for p in div.findAll('p')]
            except Exception as e:
                print(e)
                continue
            _articles[title] = content
        print(f'\n{len(_articles)} articles retrieved')
        return _articles

    def retrieve_articles(self, output_filename):
        """ retrieve articles from TC
        :param output_filename: output filename for saving articles
        """
        now = datetime.now()
        print("now =", now)
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
        print("date and time =", dt_string)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        output_filename = f'{dir_path}/{now} - {output_filename}'
        print(f'article output file = {output_filename}')
        soup = self.get_soup()
        links = self.get_all_article_links(soup)
        self.dump_json(self.extract_article_content(links), output_filename)

    @staticmethod
    def load_articles(article_filepath):
        """ load existing article data
        :param article_filepath: path to existing saved articles
        :return: articles dict
        """
        with open(article_filepath) as f:
            return json.load(f)

    @staticmethod
    def dump_json(json_data, output_filepath):
        """ dump json data to a file
        :param output_filepath: output filepath for saved data
        :param json_data: object which can be parsed to json
        """
        print(f'dumping json data to file {output_filepath}')
        with open(output_filepath, 'w') as outfile:
            json.dump(json_data, outfile, indent=2)

    @staticmethod
    def write_list_to_file(list_data, output_filepath):
        """ write a list to file line by line
        :param list_data: list data
        :param output_filepath: output filepath for saved data
        """
        with open(output_filepath, 'w') as f:
            for item in list_data:
                f.write("%s\n" % item)


# Program driver
if __name__ == '__main__':
    scraper = TechCrunchArticleScraper()
    # existing_articles = scraper.load_articles('/Users/andrewdavies/PycharmProjects/MachineLearning/nlp_topic_modelling/.........')
    # titles = existing_articles.keys()  # used when retrieving articles to ensure no duplicates are scraped
    scraper.retrieve_articles('tech_crunch_articles.json')
