#Gerekli kütüphaneleri içe aktaralım.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


#Veri setini içe aktarıyoruz.
df = pd.read_csv("NLPlabeledData.tsv", delimiter="\t", quoting=3)
#print(df.head())
#print(len(df))


#İngilizcede bulunan "the, is, are" gibi tek başına bir anlam ifade etmeyen kelimelerden veri setini temizliyoruz.
#Bu işlemi yapabilmek için kütüphaneyi indiriyoruz.
#nltk.download("stopwords")


#Şimdi veri temizleme işlemini yapalım.
#Buradaki gereksiz HTML taglerini silmek için beautifulSoap modülünü kullanacağız.
def process(review):
    sample_review = BeautifulSoup(review).get_text()
    sample_review = re.sub("[^a-zA-z]", " ", sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    swords = set(stopwords.words("english"))
    sample_review = [w for w in sample_review if w not in swords]
    return("".join(review))

train_x_tum = []
for r in range(len(df["review"])):
    if(r+1)%1000 == 0:
        print("No of reviews processed = ", r+1)
    train_x_tum.append(process(df["review"][r]))

x = train_x_tum
y = np.array(df["sentiment"])
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)

vectorizer = CountVectorizer(max_features=5000)
train_x = vectorizer.fit_transform(train_x)

train_x = train_x.toarray()

model = RandomForestClassifier(n_estimators=100)
model.fit(train_x, train_y)

test_xx = vectorizer.transform(test_x)
test_xx = test_xx.toarray()

test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(test_y, test_predict)
print(f"Doğruluk oranı: {dogruluk*100}")
