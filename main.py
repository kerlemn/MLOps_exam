from data_extraction import *


def main():
    """download data and convert to BoW"""

    # choose keywords (necessary only the first time)
    keywords = get_keywords('data/english1k.txt')

    # initialize pipeline
    pipeline = WebScrapingPipeline(keywords=keywords)

    # download Wiki pages
    while True:

        title = input('\n\npage title: ')
        if title in ('q', 'quit'):
            break

        try:
            # download text
            print(f'downloading page...\n')
            text = request_text_from_wiki(title)
            bag_of_words = text_to_bag(clean_text(text), pipeline.keywords)

            text = process_text(text)
            print(text)

            while True:
                score = input('\nscore (0 or 1): ')
                if score in ('0', '1'):
                    score = int(score)
                    break

            # add page to dataframe
            pipeline.add_page_to_df(title, bag_of_words, score)

            # save dataframe
            pipeline.dump_data()

        except ValueError:
            # page does not exist
            print(f'page {title} does not exist!')


if __name__ == '__main__':
    main()
