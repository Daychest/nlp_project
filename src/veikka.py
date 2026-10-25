from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import requests
from gensim import corpora
import pickle
import gensim


if 0:
    nltk.download('brown')
    nltk.download('wordnet_ic')
    nltk.download('gutenberg')


characterDict = dict()
characterDict["Elizabeth"] = {"a", "b", "c"}
characterDict["Darcy"] = {"a", "b", "c"}
characterDict["Jane"] = {"a", "b", "c"}

characters = {
    "Elizabeth": ["Elizabeth", "Lizzy", "Eliza"],
    "Darcy": ["Darcy"],
    "Jane": ["Jane"],
    "Bingley": ["Bingley"],
    "Wickham": ["Wickham"],
    "Lydia": ["Lydia"],
    "Mrs. Bennet": ["Mrs. Bennet"],
    "Mr. Bennet": ["Mr. Bennet"],
    "Collins": ["Collins"],
    "Charlotte": ["Charlotte"],
    "Lady Catherine": ["Lady Catherine"],
    "Kitty": ["Kitty"],
    "Mary": ["Mary"],
}


def fprint(str):
    with open("temp", "w") as f:
        print(str, file=f)


def getBookText():
    # todo: cut header?
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    response = requests.get(url)
    return response.text


def getSpeech():
    text = getBookText().replace("\r\n", "\n")
    paragraphs = text.split("\n\n")

    speech = {name: [] for name in characters}
    for p in paragraphs:
        p = " ".join(p.split())
        quotes = re.findall(r'[“"](.*?)[”"]', p)
        if not quotes:
            continue
        narration = re.sub(r'[“"].*?[”"]', " ", p)
        for name, aliases in characters.items():
            pattern = r"\b(" + "|".join(aliases) + r")\b"
            if re.search(pattern, narration):
                speech[name].extend(quotes)
                break
    return speech


def task1():
    text = " ".join(getBookText().split())
    sentences = nltk.sent_tokenize(text)

    results = []
    for name, aliases in characters.items():
        pattern = r"\b(" + "|".join(aliases) + r")\b"
        frequency = len(re.findall(pattern, text))
        citations = sum(1 for s in sentences if re.search(pattern, s))
        results.append((name, frequency, citations))

    results.sort(key=lambda r: r[1], reverse=True)

    print(f"{'Rank':<6}{'Character':<16}{'Frequency':<11}{'Citations'}")
    for i, (name, frequency, citations) in enumerate(results, 1):
        print(f"{i:<6}{name:<16}{frequency:<11}{citations}")


def task2():
    speech = getSpeech()

    # One dataframe per character + statistics
    rows = []
    for name, quotes in speech.items():
        df = pd.DataFrame({"text": quotes})
        tokens = nltk.word_tokenize(" ".join(df["text"]))
        words = [w.lower() for w in tokens if w.isalpha()]
        vocabulary = set(words)
        tags = [tag for word, tag in nltk.pos_tag(tokens)]

        rows.append({
            "Character": name,
            "Vocabulary size": len(vocabulary),
            "Tokens": len(tokens),
            "Adjectives": sum(t.startswith("JJ") for t in tags),
            "Adverbs": sum(t.startswith("RB") for t in tags),
            "Verbs": sum(t.startswith("VB") for t in tags),
            "Nouns": sum(t.startswith("NN") for t in tags),
            "Vocabulary (sample)": sorted(vocabulary)[:5],
        })

    summary = pd.DataFrame(rows).sort_values("Tokens", ascending=False)
    print(summary.to_string(index=False))


def task3():
    speech = getSpeech()
    plt.gca().set_prop_cycle(color=plt.cm.tab20.colors)

    rows = []
    for name, quotes in speech.items():
        df = pd.DataFrame({"text": quotes})
        tokens = nltk.word_tokenize(" ".join(df["text"]))
        words = [w.lower() for w in tokens if w.isalpha()]

        # vocabulary size after each token
        N, V, seen = [], [], set()
        for i, w in enumerate(words, 1):
            seen.add(w)
            N.append(i)
            V.append(len(seen))
        plt.plot(N, V, label=name)

        # Heaps' law
        x, y = np.log(N), np.log(V)
        beta, logK = np.polyfit(x, y, 1)
        y_pred = logK + beta * x
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        n = len(x)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)

        rows.append({
            "Character": name,
            "Tokens": len(words),
            "Vocabulary": len(seen),
            "K": round(np.exp(logK), 2),
            "beta": round(beta, 3),
            "R2": round(r2, 4),
            "Adj. R2": round(adj_r2, 4),
        })

    plt.xlabel("Number of tokens")
    plt.ylabel("Vocabulary size")
    plt.title("Vocabulary growth per character")
    plt.legend()
    plt.show()

    summary = pd.DataFrame(rows).sort_values("Tokens", ascending=False)
    print(summary.to_string(index=False))


def getSpeechAsTokens():
    tokenDict = dict()
    speech = getSpeech()
    for person, quotes in speech.items():
        quotesString = ' '.join(quotes)
        quotesString = quotesString.lower()
        tokens = nltk.word_tokenize(quotesString)
        tokenDict[person] = tokens
    return tokenDict


def task4():
    modalVerbs = ("can", "may", "must", "shall", "will",
                  "could", "might", "should", "would")

    speechAsTokens = getSpeechAsTokens()

    for person, tokens in speechAsTokens.items():
        xEntries = []
        yEntries = []
        tokenCount = 0
        modalVerbCount = 0

        for token in tokens:
            tokenCount += 1
            if token in modalVerbs:
                modalVerbCount += 1
            xEntries.append(tokenCount)
            yEntries.append(modalVerbCount)

        plt.plot(xEntries, yEntries, label=person)

    plt.xlabel("Number of tokens")
    plt.ylabel("Number of modal verbs")
    plt.title("Modal verbs compared to total tokens for each character")
    plt.legend()
    plt.show()


def getBookTokens():
    text = getBookText()
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return tokens


def printMatrix(matrix, rowLabels=None, columnLabels=None, decimalCount=2):
    dataFrame = DataFrame(
        matrix,
        columns=columnLabels,
        index=rowLabels
    )
    print(dataFrame.round(decimalCount))


def filterTokenList(tokenList):
    generalStopWords = stopwords.words('english')
    manualStopWords = ['must', 'one', 'shall',
                       'could', 'would', 'may', 'think', 'must', 'us', 'much']

    filteredTokens = []
    for token in tokenList:
        # Tokens have to be alphanumeric
        if not token.isalnum():
            continue
        # Tokens cannot be stopwords
        if token in generalStopWords:
            continue
        if token in manualStopWords:
            continue
        filteredTokens.append(token)

    return filteredTokens


def getSpeechAsFilteredTokens():
    speechAsTokens = getSpeechAsTokens()

    speechAsFilteredTokens = dict()
    for person, tokens in speechAsTokens.items():
        speechAsFilteredTokens[person] = filterTokenList(tokens)

    return speechAsFilteredTokens


def task5():
    tokens = getBookTokens()
    filteredTokens = filterTokenList(tokens)

    counter = Counter(filteredTokens)
    mostCommon = counter.most_common(5)

    mostCommonVector = []
    for token, count in mostCommon:
        mostCommonVector.append(token)

    vectorDict = dict()
    speechAsTokens = getSpeechAsTokens()
    for person, tokens in speechAsTokens.items():
        vectorDict[person] = [0, 0, 0, 0, 0]
        for token in tokens:
            for i, mostCommonToken in enumerate(mostCommonVector):
                if token == mostCommonToken:
                    vectorDict[person][i] += 1

    frequencyMatrix = list(vectorDict.values())

    print("Frequency of the five most frequent tokens for each character:")
    printMatrix(frequencyMatrix,
                rowLabels=vectorDict.keys(), columnLabels=mostCommonVector)
    print()

    cosineSimilarityMatrix = cosine_similarity(
        frequencyMatrix, frequencyMatrix)

    print("Cosine similarity between characters")
    printMatrix(cosineSimilarityMatrix, rowLabels=vectorDict.keys(),
                columnLabels=vectorDict.keys())


def task6():
    speechAsFilteredTokens = getSpeechAsFilteredTokens()
    people = speechAsFilteredTokens.keys()
    filteredTokenLists = list(speechAsFilteredTokens.values())

    dictionary = corpora.Dictionary(filteredTokenLists)
    corpus = [dictionary.doc2bow(text) for text in filteredTokenLists]

    topicCount = 5
    keywordCount = 10

    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, id2word=dictionary, num_topics=topicCount)
    topicStrings = ldamodel.print_topics(num_words=keywordCount)

    print("Topics:")
    for topic in topicStrings:
        print(topic)
    print()

    topics = ldamodel.show_topics(
        num_topics=ldamodel.num_topics,
        num_words=keywordCount,
        formatted=False
    )

    personTopicSimilarities = dict()
    for person in people:
        personTopicSimilarities[person] = [0, 0, 0, 0, 0]
        for topicId, words in topics:
            for word, probability in words:
                if word in speechAsFilteredTokens[person]:
                    personTopicSimilarities[person][topicId] += probability

    print("Each person's similarity with each topic:")
    printMatrix(personTopicSimilarities)
    print()

    personTopicSimilarityMatrix = list(personTopicSimilarities.values())
    cosineSimilarityMatrix = cosine_similarity(
        personTopicSimilarityMatrix, personTopicSimilarityMatrix)

    print("Topic similarity between characters using cosine similarity")
    printMatrix(cosineSimilarityMatrix, columnLabels=people, rowLabels=people)


task6()
