import json
import os
import re
from math import ceil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mpltern
import nltk
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from pandas import DataFrame
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

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


def getBookText():
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


def getSpeechAsTokens():
    tokenDict = dict()
    speech = getSpeech()
    for person, quotes in speech.items():
        quotesString = ' '.join(quotes)
        quotesString = quotesString.lower()
        tokens = nltk.word_tokenize(quotesString)
        tokenDict[person] = tokens
    return tokenDict


def printMatrix(matrix, rowLabels=None, columnLabels=None, decimalCount=2, labels=True):
    dataFrame = DataFrame(
        matrix,
        columns=columnLabels,
        index=rowLabels
    )
    print(dataFrame.round(decimalCount).to_string(header=labels, index=labels))


def filterTokenList(tokenList):
    generalStopWords = stopwords.words('english')
    manualStopWords = ['must', 'one', 'shall', 'could', 'would', 'may', 'think', 'must',
                       'us', 'much', 'well', 'know', 'never', 'see', 'good', 'oh', 'ever', 'always', 'make', 'sure', 'dear', 'indeed']

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


def task7():
    speechAsFilteredTokens = getSpeechAsFilteredTokens()
    people = speechAsFilteredTokens.keys()
    filteredTokenLists = list(speechAsFilteredTokens.values())

    topicCount = 5
    keywordCount = 10
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    # Prompt generated with help of ChatGPT
    prompt = (
        f"Find {topicCount} shared topics in these dialogue collections. "
        f"For each topic choose {keywordCount} keywords from the supplied tokens. "
        "Assign positive keyword weights summing to one within each topic. "
        'Return JSON as {"topics": [[0, [["word", 0.1], ...]], ...]}. '
        "Use topic IDs 0 to 4.\n" + json.dumps(filteredTokenLists)
    )
    # https://ai.google.dev/api/generate-content#method:-models.generatecontent
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        },
        timeout=180,
    )
    text = response.json()["candidates"][0]["content"]["parts"][-1]["text"]
    topics = json.loads(text)["topics"]
    print("Topics:")
    for topic in topics:
        print(topic)
    print()

    personTopicSimilarities = dict()
    for person in people:
        counter = Counter(speechAsFilteredTokens[person])

        personTopicSimilarities[person] = [0, 0, 0, 0, 0]
        for topicId, words in topics:
            for word, probability in words:
                if word in speechAsFilteredTokens[person]:
                    weight = probability * counter[word]
                    personTopicSimilarities[person][topicId] += weight

    print("Each person's similarity with each topic:")
    printMatrix(personTopicSimilarities)
    print()

    personTopicSimilarityMatrix = list(personTopicSimilarities.values())
    cosineSimilarityMatrix = cosine_similarity(
        personTopicSimilarityMatrix, personTopicSimilarityMatrix)

    print("Topic similarity between characters using cosine similarity")
    printMatrix(cosineSimilarityMatrix, columnLabels=people, rowLabels=people)


def getBertVector(text, tokenizer, model):
    inputs = tokenizer(text, truncation=True, return_overflowing_tokens=True)
    total = torch.zeros(model.config.hidden_size)
    count = 0

    for ids in inputs["input_ids"]:
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            result = model(input_ids=input_ids)
            token_vectors = result.last_hidden_state
            token_vectors = token_vectors[0]
            output = token_vectors[1:-1]
        total += output.sum(dim=0)
        count += len(output)

    return (total / count).numpy()


def task8(dataframes):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    vectors = []
    for _name, df in dataframes.items():
        vector = getBertVector(" ".join(df["text"]), tokenizer, model)
        vectors.append(vector)

    names = list(dataframes)
    embeddings = pd.DataFrame(vectors, index=names)
    similarity = pd.DataFrame(cosine_similarity(vectors), index=names, columns=names)
    print(f"BERT similarity:\n{similarity.round(3).to_string()}")
    return embeddings, similarity


def getSentiment(text, lemmatizer):
    tokens = []
    for token in nltk.word_tokenize(text):
        for character in token:
            if character.isalnum():
                tokens.append(token)
                break

    tagged = nltk.pos_tag(tokens)
    positive, negative, neutral = 0.0, 0.0, 0.0
    pos_tags = {"J": "a", "N": "n", "R": "r", "V": "v"}

    for word, tag in tagged:
        pos = pos_tags.get(tag[0])
        if pos is None:
            neutral += 1
            continue
        lemma = lemmatizer.lemmatize(word.lower(), pos)
        synset = lesk(tokens, lemma, pos)
        if synset is not None:
            sentiment = swn.senti_synset(synset.name())
            positive += sentiment.pos_score()
            negative += sentiment.neg_score()
            neutral += sentiment.obj_score()
        else:
            neutral += 1

    if not tokens:
        return [0.0, 0.0, 1.0]
    return [positive / len(tokens), negative / len(tokens), neutral / len(tokens)]


def task10(dataframes):
    lemmatizer = WordNetLemmatizer()
    results = []
    rows = ceil(len(dataframes) / 3)
    fig = plt.figure(figsize=(12, rows * 3.5), layout="constrained")

    for i, (name, df) in enumerate(dataframes.items(), 1):
        scores = []
        for message, text in enumerate(df["text"], 1):
            positive, negative, neutral = getSentiment(text, lemmatizer)
            scores.append([positive, negative, neutral])
            results.append([name, message, text, positive, negative, neutral])

        scores = pd.DataFrame(scores, columns=["Positive", "Negative", "Neutral"])
        ax = fig.add_subplot(rows, 3, i, projection="ternary")
        ax.scatter(scores["Positive"], scores["Negative"], scores["Neutral"], s=12, alpha=0.5)
        ax.set_tlabel("Positive")
        ax.set_llabel("Negative")
        ax.set_rlabel("Neutral")
        ax.set_title(name)
        ax.grid(alpha=0.3)

    results = pd.DataFrame(results, columns=["Character", "Message", "Text", "Positive", "Negative", "Neutral"])
    print(f"Sentiment scores:\n{results.drop(columns='Text').round(3)}")
    plt.show()
    return results


def main():
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("wordnet")
    nltk.download("sentiwordnet")

    dataframes = {}
    for name, quotes in getSpeech().items():
        if quotes:
            dataframes[name] = pd.DataFrame({"text": quotes})

    task7()
    task8(dataframes)
    task10(dataframes)


if __name__ == "__main__":
    main()
