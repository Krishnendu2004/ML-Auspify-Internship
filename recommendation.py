import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('Dataset.csv')
print(df.head())
print("Number of rows and columns:", df.shape)
print("No:of Missing values in the dataset:",df.isnull().sum())

df=df.drop_duplicates(subset=["title", "type", "release_year"])
df=df.reset_index(drop=True)
print("Number of rows and columns after removing duplicates:", df.shape)
print(df.head())

#STEP 1: Prepare content-related features
movies = len(df[df["type"] == "Movie"])
tv_shows = len(df[df["type"] == "TV Show"])

print("Number of Movies:", movies)
print("Number of TV Shows:", tv_shows)

plt.bar(["Movies", "TV shows"], [movies, tv_shows], color=["red", "green"])
plt.title("Movies vs TV Shows")
plt.xlabel("No:of titles")
plt.savefig("Movies_vs_tvshows.png")
plt.show()

df["director"]=df["director"].replace("Not Given"," ")
df["event"] = (df["listed_in"] + " " + df["listed_in"] + " " +df["director"] + " " + df["country"] + " " + df["rating"])
print(df.head())

#STEP 2:  Convert text data into machine-readable format
tfid = TfidfVectorizer()
matrix = tfid.fit_transform(df["event"])
print("Matrix shape:", matrix.shape)

genres = []                                  
for text in df["listed_in"]:
    genres.extend(text.split(", "))           

names = ["International Movies", "Dramas", "Comedies", "International TV Shows",
         "Documentaries", "Action & Adventure", "TV Dramas", "Independent Movies",
         "Children & Family Movies", "Romantic Movies"]
count = []
for name in names:
    count.append(genres.count(name))         

plt.barh(names, count,height=0.5, color="blue")
plt.title("Popular 10 Generes")
plt.xlabel("No:of titles")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("different_genres.png")
plt.show()

#STEP 3: Calculate content similarity scores
def recommendation(title):
    row = df[df["title"] == title].index[0]
    scoring = cosine_similarity(matrix[row], matrix).flatten()
    scoring[row] = -1
    best = np.argsort(scoring)[::-1][:10]
    return df.loc[best, ["title", "type", "listed_in"]]

#STEP 4: Generate recommendations for selected titles
for name in ["Narcos", "Midnight Mass", "Ganglands"]:
    print("\nbecause you watched", name + ":\n", recommendation(name))

#STEP 5: Evaluate recommendation quality
np.random.seed(42)      
good = 0                
rand = 0         
total = 0               

for row in np.random.choice(len(df), 100, replace=False):
    scores = cosine_similarity(matrix[row], matrix).flatten()
    scores[row] = -1
    best = np.argsort(scores)[::-1][:10]                    
    random_titles = np.random.choice(len(df), 10, replace=False)   

    mygenres = set(df["listed_in"][row].split(", "))       

    for i in best:
        if len(mygenres & set(df["listed_in"][i].split(", "))) > 0:  
            good = good + 1
    for i in random_titles:
        if len(mygenres & set(df["listed_in"][i].split(", "))) > 0:
            rand =rand + 1
    total = total + 10

modelaccu = good / total
randomaccur = rand / total
print("\nModel accuracy   :", modelaccu)
print("Random picks accuracy:", randomaccur)

plt.bar(["Model", "Random Ones"], [modelaccu, randomaccur], color=["blue", "violet"])
plt.title("Recommendation quality")
plt.ylabel("Accuracy")
plt.savefig("Model Strength.png")
plt.show()