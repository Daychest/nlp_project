import requests


if 0:
    import nltk
    nltk.download('brown')
    nltk.download('wordnet_ic')
    nltk.download('gutenberg')


def getBookText():
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    response = requests.get(url)
    return response.text


def task6():
    print(getBookText())
    pass


task6()
