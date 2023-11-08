import requests


WIKI_URL = 'https://en.wikipedia.org/w/api.php'
REMOVE_CHARS = '[]{}()<>|\\^~`@#$%&*_-+=;:\'",.?/!\t\n'


def clean_text(text: str):
    """ clean text from html tags and other stuff """
    # remove special chars
    for c in REMOVE_CHARS:
        text = text.replace(c, ' ')

    # split
    v = [x for x in text.split(' ') if x != '']
    return v


class WebScrapingPipeline:

    def __init__(self, wiki_plain_text_dir_path, wiki_bags_dir_path, bag_creator):
        self.wiki_plain_text_dir_path = wiki_plain_text_dir_path
        self.wiki_bags_dir_path = wiki_bags_dir_path
        self.bag_creator = bag_creator

    def request_to_wiki(self, title):
        """ return response from wikipedia api given some url and title """
        params = {'action': 'query',
                  'format': 'json',
                  'titles': title,
                  'prop': 'extracts',
                  'exintro': True,
                  'explaintext': True}
        return requests.get(WIKI_URL, params=params).json()

    def request_text_from_wiki(self, title):
        """ extract text from wikipedia page given some title """
        return self.request_to_wiki(title)['query']['pages']['23862']['extract']

    def convert_text_to_bag(self, text: str):
        """return bag of words given some text"""
        text = clean_text(text)
        return self.bag_creator(text)

    def save_bag(self, title, bag):
        """"""
        with open(f'{self.wiki_bags_dir_path}/{title}.txt', 'w') as f:
            f.write(str(bag))


def test_1(title='Python (programming language)'):
    """ test request_text_from_wiki """
    words = ['Python', 'programming', 'gigachad', 'meme']

    def bag_creator(text):
        return [word in text for word in words]

    # initialize pipeline
    pipeline = WebScrapingPipeline('data/wiki_plain_text', 'data/wiki_bags', bag_creator)

    # get text
    text = pipeline.request_text_from_wiki(title)

    # get and save bag
    bag = pipeline.convert_text_to_bag(text)
    pipeline.save_bag(title, bag)

    print(text)
    print(bag)


if __name__ == '__main__':
    test_1()
