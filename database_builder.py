"""
This class is used to pre-process the wikipedia .parquet data
"""
import pandas as pd
import os
import numpy as np
from misc import clean_text
from time import time


class DatabaseBuilder:
    """

    """
    def __init__(self, path):
        self.path = path
        self.df = pd.read_parquet(path)
        self.values = None

    def save_data(self, path):
        """save data in a parquet file"""
        print('saving...')
        self.df.to_parquet(path, index=False)

    def _build_matrix(self, keywords, epoch_print=20_000):
        """build the binary matrix"""
        shape = (len(self.df), len(keywords))
        print(f'building boolean matrix {shape}')
        self.values = np.full(shape, False)
        for i in range(len(self.df)):
            if i % epoch_print == 0:
                print(i)
            text = self.df.loc[i, 'TEXT']
            text = clean_text(text)
            v = text.split(' ')
            s = set(v)
            for j in range(len(keywords)):
                if keywords[j] in s:
                    self.values[i, j] = True

    def _add_features_to_df(self, keywords):
        """

        """
        print('adding features...')
        df = pd.DataFrame(self.values, columns=keywords)
        self.df = pd.concat([self.df, df], axis=1)

    def run(self, keywords, new_file_path, epoch_print=20_000):
        """
        create a new version of the dataset
        """
        # rename columns to uppercase
        self.df.columns = self.df.columns.str.upper()

        self._build_matrix(keywords, epoch_print)
        self._add_features_to_df(keywords)
        self.save_data(new_file_path)


def main():
    """
    preprocess wikipedia database
    """
    t0 = time()

    # paths
    dir_path = 'data\\Wikipedia_full_archive'
    new_dir_path = 'data\\database'
    kw_path = 'data\\english10k.txt'

    # names of parquet files
    file_names = list(os.listdir(dir_path))
    file_names = [name for name in file_names if not name.startswith('wiki')]
    file_names.sort(reverse=True)

    # list of english keywords
    with open(kw_path, 'r') as f:
        keywords = list(f.read().split('\n'))

    # iteratively process all files
    for name in file_names:
        print(f'\n>>> processing {name}')
        path = f'{dir_path}\\{name}'
        new_path = f'{new_dir_path}\\{name}'

        if os.path.isfile(new_path):
            print('requirement already satisfied!')
        else:
            db = DatabaseBuilder(path)
            db.run(keywords, new_path)

            t1 = time() - t0
            print(f'total time: {t1//60:.0f}m {t1%60:.1f}s')


def test():
    path = 'data\\database\\z.parquet'
    db = DatabaseBuilder(path)
    print(db.df.head())

    for name in db.df.keys():
        print(name)


if __name__ == '__main__':
    # main()
    test()
