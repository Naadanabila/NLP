import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import ast
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Streamlit App
st.title("Prediksi Subjek Makalah Penelitian")
st.write("Aplikasi berbasis web untuk memprediksi (mengklasifikasi) subjek dari Research Paper berbahasa Inggris")

# Load the image
img = Image.open('icon.png')
img = img.resize((400, 400))

# Center the image using CSS and HTML
st.markdown(
    f"""
    <style>
        .center {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
    </style>
    <div class="center">
        <img src="data:image/png;base64,{image_to_base64(img)}" alt="Centered Image" width="400" height="400">
    </div>
    """,
    unsafe_allow_html=True
)

# Input user
title_input = st.text_input("Masukkan Judul Research Paper:")
abstract_input = st.text_area("Masukkan Abstrak Research Paper:")

# Prediksi subjek
if st.button("Prediksi"):
    # Baca dataset
    train = pd.read_csv('paper2019.csv')

    # Mengonversi string representasi list ke list
    for i in range(len(train)):
        x = train.Subjects[i]
        x = ast.literal_eval(x)
        x = [n.strip() for n in x]
        train.Subjects[i]=x


    # One Hot Encoding dari Subjects
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    train = train.join(pd.DataFrame(mlb.fit_transform(train.pop('Subjects')),
                            columns=mlb.classes_,
                            index=train.index))

    # train.drop([15015],inplace=False)

    # Case Folding
    def clean_text(text):
        # remove backslash-apostrophe
        text = text.replace("\\n", " ")
        # remove everything except alphabets
        text = re.sub("[^a-zA-Z]"," ",text)
        # remove whitespaces
        text = ' '.join(text.split())
        # convert text to lowercase
        text = text.lower()
        text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)
        return text

    train['Abstract'] = train['Abstract'].apply(lambda x: clean_text(x))
    train['Title'] = train['Title'].apply(lambda x: clean_text(x))

    def freq_words(x, terms = 30):

        all_words = ' '.join([text for text in x])
        all_words = all_words.split()
        fdist = nltk.FreqDist(all_words)
        words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
        d = words_df.nlargest(columns="count", n = terms)


    freq_words(train['Abstract'], 100)

    # Menghapus stopwords
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(text):
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    train['Abstract'] = train['Abstract'].apply(lambda x: remove_stopwords(x))

    def freq_words(x, terms):

        all_words = ' '.join([text for text in x])
        all_words = all_words.split()
        fdist = nltk.FreqDist(all_words)
        words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
        d = words_df.nlargest(columns="count", n = terms)

    freq_words(train['Abstract'], 100)

    # Grouping by month
    import datetime
    temp = []
    for i in range(len(train)):
        a = train.Date[i]
        datee = datetime.datetime.strptime(a, "%Y-%m-%d")
        temp.append(datee.month)

    train['Month'] = temp

    train.groupby(['Month']).sum()

    # Menggabungkan judul dan abstrak
    train['abstract_new'] = train['Abstract'] + ' ' + train['Title']

    # Stemming of Vocabulary
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    ps = PorterStemmer()

    nltk.download('punkt')

    x = train.iloc[:,82]
    x = x.apply(lambda y:word_tokenize(y))
    x = x.apply(lambda x: [ps.stem(y) for y in x])

    sent=[]
    for row in x:
        sequ=''
        for word in row:
            sequ=sequ +' ' + word
        sent.append(sequ)
    x=sent

    train['abstract_new']=x

    # Applying tf-idf weights
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

    traintfidf=tfidf_vectorizer.fit_transform(train.iloc[:,82].values)
    train_new = traintfidf.todense()

    # Split data
    xtrain, xval, ytrain, yval = train_test_split(train_new, train.iloc[:, 4:81], test_size=0.2, random_state=9)

    # Model SVM
    clf = OneVsRestClassifier(LinearSVC())
    xtrain_array = np.asarray(xtrain)
    ytrain_array = np.asarray(ytrain.values)

    # Training model
    clf.fit(xtrain_array, ytrain_array)
    # Membersihkan input
    cleaned_abstract = clean_text(abstract_input)
    cleaned_title = clean_text(title_input)
    combined_text = cleaned_abstract + ' ' + cleaned_title

    # Stemming
    tokenized_text = word_tokenize(combined_text)
    stemmed_text = [ps.stem(word) for word in tokenized_text]
    preprocessed_text = ' '.join(stemmed_text)

    # TF-IDF
    tfidf_input = tfidf_vectorizer.transform([preprocessed_text]).todense()

    # Prediksi
    tfidf_input_array = np.asarray(tfidf_input)
    prediction = clf.predict(tfidf_input_array)

    # Inverse transform to get subject names
    predicted_subjects = mlb.inverse_transform(prediction)

    # Flatten the list of subjects
    flat_subjects = [subject for subjects in predicted_subjects for subject in subjects]

    # Tampilkan hasil
    st.write('Hasil Prediksi : ')
    for subject in flat_subjects:
        st.markdown(f'<p style="background-color:#673b8d; color:white; padding: 10px; border-radius: 5px;">{subject}</p>', unsafe_allow_html=True)

    
    
