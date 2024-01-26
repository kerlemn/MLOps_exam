"""
This class is used to interact with the database of all the Wikipedia pages

--------------------
Useful methods:
- get_random_pages: select a random file of the database, then get a list of random titles from that file
- get_training_data: given a list of titles, get a dataframe with the corresponding binary arrays
"""
import pandas as pd
import numpy as np
import os


class WikiDatabase:

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.file_names = [s for s in os.listdir(dir_path) if s.endswith('.parquet')]
        self.n_files = len(self.file_names)

    def get_random_pages(self, n_pages):
        """select a random file of the database, then get a list of random titles from that file"""
        file_name = self.file_names[np.random.randint(self.n_files)]
        df = pd.read_parquet(os.path.join(self.dir_path, file_name))
        return df.sample(n_pages).title.values.tolist()

    def get_training_data(self, titles):
        """given a list of titles, get a dataframe with the corresponding binary arrays"""
        data = pd.DataFrame()
        data = data.reset_index(drop=True)
        for file_name in self.file_names:
            df = pd.read_parquet(os.path.join(self.dir_path, file_name))
            df = df[df.TITLE.isin(titles)]
            df = df.reset_index(drop=True)
            data = pd.concat([data, df])
        return data
