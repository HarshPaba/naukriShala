
import re
from ftfy import fix_text
import numpy as np
import pandas as pd
import streamlit as st 
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords


data = pd.read_csv(r"https://raw.githubusercontent.com/OmkarPathak/pyresparser/master/pyresparser/skills.csv") 
SKILLS_DB = list(data.columns.values)

stopw  = set(stopwords.words('english'))
df=pd.read_csv(r"https://raw.githubusercontent.com/HarshPaba/naukri_Shala/main/job_final.csv")
df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    found_skills = set()

    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)

    return found_skills


def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def builder(data):
    skills=[]
    skills.append(' '.join(word for word in data))

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)

    
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    test = (df['test'].values.astype('U'))

    def getNearestN(query):
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances, indices


    distances, indices = getNearestN(test)
    test = list(test) 
    matches = []

    for i,j in enumerate(indices):
        dist=round(distances[i][0],2) 
        temp = [dist]
        matches.append(temp)
        
    matches = pd.DataFrame(matches, columns=['Match confidence'])

    df['match']=matches['Match confidence']
    df1=df.sort_values('match')
    df1=df1[['Position','Company','Location','url']].head(10).reset_index()
    df1=df1.drop('index',axis=1)
    return df1

def main():
    st.title("NaukriShala")

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    
    if uploaded_file is not None:
        if st.button("Predict"):
            extracted_text=extract_text(uploaded_file)
            skills=extract_skills(extracted_text)
            st.text("Skills extracted succesfully !!!!")
            st.subheader("Here are some jobs based on your skills")
            df=builder(skills)
            
            st.table(df)

if __name__=='__main__':
    main()