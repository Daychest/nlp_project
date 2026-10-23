import re
import requests
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nltk.download('brown')
nltk.download('wordnet_ic')
nltk.download('gutenberg')
nltk.download('punkt_tab')


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
    # Elizabeth puheliain.  Puhuu n. 3 kertaa enemmän kun Darcy. Omistaa myös isoimman sanaston. B arvo on pienin, eli uusia sanoja tulee pitkässä puheessa suhteessa harvoin.
    # Darcy, Jane ja Bingley puhuvat kohtalaisesti. Darcyllä on suurempi B kuin Janella, joten hän käyttää vaihtelempaa sanastoa.
    # Mrs Bennet ja Lydia puheliaitam, mutta sanasto on suhteessa suppea, koska matalaat B arvot ja heikommat sovitukset. he puhuvat paljon, mutta sanasto on suppea.
    # Mr Bennet, Collins ja lady catherine puhuvat vähän, mutta vaihtelevasti (B 0.84). 
    # Mary ja Kitty puhuvat vähän. Korkea B arvo johtuu luultavasti pienestä aineistosta, koska alussa jokainen sana on uusi. 

task1()
task2()
task3()
