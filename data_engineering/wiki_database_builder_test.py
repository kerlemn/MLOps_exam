import matplotlib.pyplot as plt
from wiki_database_builder import *
import numpy as np
from time import time


def test_text_to_bag_of_words():
    text = 'Hello, hello world! 2023'
    allowed_chars = WikiDatabaseBuilder.KEEP_CHARS
    print(text_to_bag_of_words(text, allowed_chars))


def test_parquet():
    file_path = 'data\\wiki_bow\\0.parquet'
    df = pd.DataFrame(columns=['TITLE', 'SCORE'])
    df.to_parquet(file_path, index=False, engine='fastparquet')


def test_parquet_performance(n=100):
    """
    create a square binary matrix of increasing size and store it as parquet,
    then plot the time it takes to read and write
    """
    read_times = []
    write_times = []

    x_ = [(i+1)**2 for i in range(n)]

    for i in x_:
        matrix = np.random.randint(0, 2, size=(i, i))

        # use indices as column names
        df = pd.DataFrame(matrix, columns=[str(i) for i in range(i)])

        file_path = f'data\\test\\{i}.parquet'

        t = time()
        df.to_parquet(file_path, index=False, engine='fastparquet')
        write_times.append(time() - t)

        t = time()
        pd.read_parquet(file_path)
        read_times.append(time() - t)

    plt.title('Parquet Performance')
    plt.scatter(x_, read_times, label='read')
    plt.scatter(x_, write_times, label='write')
    plt.xlabel('matrix size')
    plt.ylabel('time (s)')
    plt.legend()
    plt.show()


def test_building_dataset(n_files=1000):
    """ test the class WikiDatabaseBuilder """
    wiki_path = 'data\\wiki_data'
    new_path = 'data\\wiki_bow'
    keywords_path = 'data\\keywords1k.txt'
    builder = WikiDatabaseBuilder(wiki_path, new_path, n_files=n_files, keyword_path=keywords_path)
    builder.run()


def test_read_dataset():
    """ test reading the dataset """
    new_path = 'data\\wiki_bow'

    for file_name in os.listdir(new_path):
        file_path = os.path.join(new_path, file_name)
        df = pd.read_parquet(file_path)
        print()
        print(file_name)
        print(df)
        print(len(df))



if __name__ == '__main__':
    # test_text_to_bag_of_words()
    # test_parquet()
    # test_parquet_performance()

    # test_building_dataset()
    test_read_dataset()
