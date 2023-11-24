

KEEP_CHARS = 'abcdefghijklmnopqrstuvwxyz' + ' '


def clean_text(text: str, encoding='utf-8') -> str:
    """ clean text from html tags and other stuff """

    # utf-8
    b = text.encode(encoding, errors="replace")
    text = b.decode(encoding)

    # lower case
    text = text.lower()

    # keep only some chars
    text = ''.join([x if (x in KEEP_CHARS) else ' ' for x in text])

    return text


def text_to_bag(text, keywords):
    """convert text to bag of words"""
    return [word in text for word in keywords]


def get_keywords(path) -> list:
    """ return the list of keywords from file """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        return [x for x in text.split('\n') if x]


def process_text(s, n=5, width=80):
    """cleans and extracts only the first chapter of the text"""
    s = s.replace('  ', '')
    v = s.split('\n')
    v = [x for x in v if len(x) != 1]
    s = '\n'.join(v)
    v = s.split('\n\n')[:n]
    s = '\n'.join(v)
    i = 0
    j = 0
    while i < len(s):
        j += 1
        if s[i] == '\n':
            j = 0
        if j == width:
            s = s[:i] + '-\n' + s[i:]
            j = 0
        i += 1
    return s


def moderate_vocabulary():
    """remove unwanted words from vocabulary"""

    # load vocabulary
    with open('data/english10k_full.txt', 'r') as f:
        words = f.read().split('\n')

    # load words to remove from vocabulary
    with open('data\\moderated.txt', 'r') as f:
        remove = f.read().split('\n')

    # remove words and save
    words = [w for w in words if w not in remove]
    with open('data/english10k.txt', 'w') as f:
        txt = '\n'.join(words)
        f.write(txt)
