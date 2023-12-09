"""
This class is used to create the database of words contained in each Wikipedia page.
Each file contains a list of titles and a bag of words for each title.
Each file of the dataset is a parquet file that contains random titles.
The dataframe has the following columns:
- TITLE: the title of the wikipedia page
- <keyword1>: is <keyword1> contained in the page?
- <keyword2>: is <keyword2> contained in the page?
...
- <keywordN>: is <keywordN> contained in the page?

The keywords are taken from a txt file.

"""
import hashlib
import os
import pandas as pd


def get_file_name(title, n, ext='parquet'):
    """return the page id from the title as an integer mod n"""
    enc = title.encode('utf-8')
    h = hashlib.sha1(enc).hexdigest()
    return f'{int(h, 16) % n}.{ext}'


def text_to_bag_of_words(text, allowed_chars: str) -> set:
    """
    convert text to BoW
    lower case -> remove non alphabetic characters -> split
    """
    v = [x if (x in allowed_chars) else ' ' for x in text.lower()]
    text = ''.join(v)
    return {x for x in text.split(' ') if x != ''}


def binary_array(bow, keywords):
    """return a binary array that indicates if a keyword is contained in the BoW"""
    return [1 if x in bow else 0 for x in keywords]


class WikiDatabaseBuilder:

    KEEP_CHARS = 'abcdefghijklmnopqrstuvwxyz 0123456789'
    SPECIAL_COLUMNS = ['TITLE']

    def __init__(self, wiki_path, new_path, n_files, keyword_path=None):
        self.wiki_path = wiki_path
        self.new_path = new_path
        self.n_files = n_files
        self.keyword_path = keyword_path
        self.keywords = None

        if self.keyword_path is not None:
            self.load_keywords(self.keyword_path)
        self._build_env()

    def load_keywords(self, path):
        """set the keywords from a txt file"""
        with open(path, 'r') as f:
            self.keywords = [x.strip() for x in f.readlines()]

    def _build_env(self):
        """ create the new directories and files that don't exist """

        # create directory if it doesn't exist
        if not os.path.exists(self.new_path):
            os.mkdir(self.new_path)

    def build_file(self, file_name, file_path, wiki_files, print_epoch=100_000):
        """ build a file of the dataset """

        print(f'\n>> file_name {file_name}')

        # create empty dataframe
        columns = self.SPECIAL_COLUMNS + self.keywords
        df = pd.DataFrame(columns=columns)

        # read the wiki files
        for wiki_file in wiki_files:

            print(f'>>> reading from {wiki_file}')

            # read the file
            wiki_file_path = os.path.join(self.wiki_path, wiki_file)
            wiki_df = pd.read_parquet(wiki_file_path)

            # iterate over the rows
            for i, row in wiki_df.iterrows():
                if i % print_epoch == 0:
                    print(f'{i}/{len(wiki_df)}')
                title = row['title']

                # compute the id corresponding to the title
                title_file_name = get_file_name(title, self.n_files)

                if title_file_name == file_name:
                    text = row['text']
                    bow = text_to_bag_of_words(text, self.KEEP_CHARS)
                    binary = binary_array(bow, self.keywords)
                    df.loc[len(df)] = [title] + binary

            print(f'({len(df)} rows)')

        print(f'>> saving to file {file_name} ({len(df)} rows)')
        df.to_parquet(file_path, index=False, engine='fastparquet')
        print(f'>> file {file_name} created')

    def run(self):
        """ run dataset builder
        Read the wiki files one by one, then convert each page to bag of words and save it
        in a pseudo-random file with get_file_name()
        """
        # find list of Wikipedia files
        wiki_files = os.listdir(self.wiki_path)
        # drop wiki_2023_index.parquet
        wiki_files.remove('wiki_2023_index.parquet')
        wiki_files = sorted([x for x in wiki_files if x.endswith('.parquet')], reverse=True)
        print(f'reading from: {wiki_files}\n')

        # iterate over file ids
        for i in range(self.n_files):
            file_name = f'{i}.parquet'
            file_path = os.path.join(self.new_path, file_name)

            # if file doesn't exist
            if not os.path.exists(file_path):
                self.build_file(file_name, file_path, wiki_files)
        print('\nDone!')
