import pandas as pd
import kagglehub
import os
import sys
import shutil
from  termcolor import colored
import string
import re
import sys
from collections import Counter, defaultdict
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


agg_methods = {'1':"Federated Averaging (FedAvg)", '2':"Federated Matched Averaging (FedMA)", '3':"All Model Averaging (AMA)", '4': "One Model Selection (OMS)", '5':"Best Models Averaging (BMA)", '6': "FedProx", '7': "Hybrid Approaches"}
models = {'1': 'Linear Regression', '2': 'Logistic Regression', '3': 'Mutli-Layer Perceptron (MLP)', '4': 'Long-Short Term Memory (LSTM)'}
all_models = { '1': LinearRegression(), '2': LogisticRegression(random_state=16) ,'3': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500), '4': Sequential() }
regions = {'1': 'Africa', '2': 'America', '3': 'Middle east', '4': 'Europe', '5': 'Asia', '6': 'World Wide', '7': 'Non-classified'}
cats = {'1': 'Stock Prices', '2': 'News Sentiment', '3': 'Other (You will specify the features later)' }
markets = {'1': 'Technology Market', '2': 'Blue-Chip Market', '3': 'Emerging Markets', '4': 'Energy & Oil Market', '5': 'Financial Market (Banking and Insurance)', '6': 'Healthcare & Pharmaceutical Market', '7': 'Consumer Goods & Retail Market', '8':'Industrial & Manufacturing Market', '9':'Real Estate Market (REITs)', '10':'Telecommunications Market', '11':'Cryptocurrency & Blockchain Market'}
markets_examples = {'1': 'NASDAQ (USA)', '2': 'Dow Jones Industrial Average (DJIA) (USA), FTSE 100 (UK)', '3':'Nifty 50 (India), Shanghai Stock Exchange (China), Bovespa (Brazil)', '4':'S&P Global Energy Index, NYSE Arca Oil Index (XOI)', '5':'S&P Financials Index, KBW Bank Index (BKX)', '6':'NYSE Healthcare Index, NASDAQ Biotechnology Index', '7':'S&P Consumer Discretionary Index, NYSE Retail Index', '8':'Dow Jones Transportation Index, S&P Industrials Index', '9':'S&P Real Estate Index, FTSE NAREIT Equity REITs Index', '10':'S&P Communications Index, NYSE Telecom Index', '11':'NASDAQ Crypto Index, Coinbase Stock (COIN), Bitcoin ETFs'}
markets_details = {'1': 'Includes companies in software, hardware, semiconductors, cloud computing, AI, and cybersecurity.', '2': 'Composed of well-established, financially stable companies with a long track record.', '3': 'Includes stocks from developing countries with high growth potential.', '4': 'Focused on oil, gas, renewable energy, and utilities.', '5': 'Covers banking, asset management, fintech, and insurance companies.', '6': 'Includes biotech, pharmaceuticals, hospitals, and medical device companies.', '7': 'Includes luxury brands, fast-moving consumer goods (FMCG), and e-commerce.', '8':'Includes aerospace, defense, transportation, construction, and heavy machinery.', '9':'Composed of companies investing in real estate properties and development.', '10':'Covers internet providers, mobile network operators, and satellite communications.', '11':'Includes companies involved in crypto exchanges, blockchain technology, and DeFi.'}
markets_companies = {'1': 'Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Google (GOOGL), Amazon (AMZN),...', '2': 'Coca-Cola (KO), Johnson & Johnson (JNJ), IBM, McDonald\'s (MCD), Procter & Gamble (PG),...', '3':'Reliance Industries (India), Alibaba (China), Vale (Brazil), Tata Motors (India),...', '4':'ExxonMobil (XOM), Chevron (CVX), BP, Shell, Saudi Aramco,...', '5':'JPMorgan Chase (JPM), Goldman Sachs (GS), Wells Fargo (WFC), Visa (V), PayPal (PYPL),...', '6':'Pfizer (PFE), Moderna (MRNA), Johnson & Johnson (JNJ), Merck (MRK), Roche (ROG),...', '7':'Walmart (WMT), Amazon (AMZN), Nike (NKE), Procter & Gamble (PG), Tesla (TSLA).', '8':'Boeing (BA), Caterpillar (CAT), Lockheed Martin (LMT), General Electric (GE),...', '9':'Simon Property Group (SPG), Prologis (PLD), Public Storage (PSA),...', '10':'AT&T (T), Verizon (VZ), T-Mobile (TMUS), Vodafone (VOD),...', '11':'Coinbase (COIN), MicroStrategy (MSTR), Bitcoin ETFs (BITO, IBIT), Riot Blockchain (RIOT),...'}

# nltk.download('stopwords')

def load_dataset(file_path):
    # Get the file extension
    l, file_extension = os.path.splitext(file_path)

    # Load dataset based on the file extension
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx' or file_extension == '.xls':
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    elif file_extension == '.html':
        return pd.read_html(file_path)[0]  # Return the first table
    elif file_extension == '.sas7bdat':
        return pd.read_sas(file_path)
    elif file_extension == '.parquet':
        return pd.read_parquet(file_path)
    elif file_extension == '.pkl' or file_extension == '.pickle':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def download_dataset(default_path="/home/donifaranga/datasets"):

    sites = {'1':'kaggle'}

    print("--------------------------------------")
    print(" GET YOUR DATASET")
    print("--------------------------------------")

    print('|> Please select the repository:\n')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in sites.items():
        print('\t',index,". "+name)

    print('\n\t|>> ',end="")
    site_index = input("").strip()
    site = sites[site_index] if site_index in sites.keys() else "kaggle"
    print(colored(f"\t|>> {site} \n",'blue'))

    print('|> Provide the dataset repository link:')
    print('\n\t|>> ',end="")
    data_link = input("").strip()
    data_link = data_link if data_link != "" else "mayankanand2701/tesla-stock-price-dataset"
    print(colored(f"\t|>>  {data_link}\n",'blue'))

    print('|> Provide the full directory path where to save the downloaded dataset:')
    print('\n\t|>> ',end="")
    dataset_path = input("").strip()
    dataset_path = dataset_path if dataset_path != "" else default_path
    print(colored(f"\t|>>  {dataset_path}\n",'blue'))

    print('|> Please specify the market categorie on which the dataset is based on:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in markets.items():
        print('\t',index,". "+name)
        print('\t\t> ',markets_details[index])
        # print('\t\t> ',markets_companies[index])
    print('\n\t|>> ',end="")
    market_index = input("").strip()
    market_index = market_index if market_index in markets.keys() else "1"
    print(colored(f"\t|>> {markets[market_index]}\n", 'blue'))

    print('|> Please specify the region on which the dataset is based on:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in regions.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    region_index = input("").strip()
    region_index = region_index if region_index in regions.keys() else "1"
    print(colored(f"\t|>> {regions[region_index]}\n", 'blue'))

    print('|> Specify the type or categorie of the dataset (stock price data, feedback data,...):')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in cats.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    cat_index = input("").strip()
    cat = cats[cat_index] if cat_index in cats.keys() else "General"
    print(colored(f"\t|>> {cat}\n", 'blue'))

    source_dataset = os.path.join(dataset_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower())

    if not os.path.exists(source_dataset):
        os.makedirs(source_dataset)


    if site == 'kaggle':
        try:
            path = kagglehub.dataset_download(data_link)
            print(colored('|> Dataset downloaded successfully!','green'))
            print('----------------------------------------')
            if os.path.isdir(path) == False:
                file_name = os.path.basename(path)
                shutil.copyfile(path,os.path.join(source_dataset,file_name))
                print('\t|> Path: ', path)
                os.remove(path)
            else:
                for i,file in enumerate(os.listdir(path)):
                    file_path = os.path.join(path, file)
                    final_path = os.path.join(source_dataset,file)
                    if os.path.isfile(file_path):
                        shutil.copyfile(file_path,final_path)
                        file_name = os.path.basename(final_path)
                        print("\t|> File[",i+1,"]: ", file_name)

        except Exception as e:
            print(colored(f"|> Error while downloading the dataset: {e}",'red'))

    print('\n---------------[END]\n')

def show_performances():

    print("-------------------------------------------------------------")
    print(" FL-FRAMEWORK MODELS PERFORMANCES")
    print("-------------------------------------------------------------")

    print('|> Please specify the region on which the dataset is based on:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in regions.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    region_index = input("").strip()
    region_index = region_index if region_index in regions.keys() else "1"
    print(colored(f"\t|>> {regions[region_index]}\n", 'blue'))


    print('|> Please specify the type or categorie of the dataset to use (stock price data, feedback data,...):')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in cats.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    cat_index = input("").strip()
    cat_index = cat_index if cat_index in cats.keys() else "1"
    print(colored(f"\t|>> {cats[cat_index]}\n", 'blue'))


    print('|> Select the model for training:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in models.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    model_index = input("").strip()
    model_index = model_index if model_index in models.keys() else "1"
    print(colored(f"\t|>> {models[model_index]}\n", 'blue'))

    print('|> Select the federated learning aggregate method:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in agg_methods.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    agg_index = input("").strip()
    agg_index = agg_index if agg_index in agg_methods.keys() else "1"
    print(colored(f"\t|>> {agg_methods[agg_index]}\n", 'blue'))



    df = pd.read_csv('models_performances.csv')

    print(df)


    filtered_data = df[df['model'] == model_index]

    print("|> Performances:")
    print(filtered_data)

    # Step 3: Plotting the accuracy and error
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(data=filtered_data, x='client', y='accuracy', palette='viridis')
    plt.title(f'Accuracy of Clients for {models[model_index]} using {agg_methods[agg_index]}')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')

    # Plot Error
    plt.subplot(1, 2, 2)
    sns.barplot(data=filtered_data, x='client', y='error', palette='viridis')
    plt.title(f'Error of Clients for {models[model_index]} using {agg_methods[agg_index]}')
    plt.xlabel('Client ID')
    plt.ylabel('Error')

    plt.tight_layout()
    plt.show()

    print()



def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def cosine_similarities(vector_a,vectors):
    similarities = []
    for vector in vectors:
        similarities.append(cosine_similarity(vector_a,vector))
    return similarities


def clean_text(text:str, tokens=[], lowercase=True, no_stopwords=True, no_digits=True, no_urls=True, no_puncts=True):

    if lowercase:
        text = text.lower()

    if no_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9][$-_@.&+]|[!*\\(\\),]|(?:[$-_@.&+])|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)

    if no_puncts:
        translator = str.maketrans('','',string.punctuation)
        text = text.translate(translator)

    if no_digits:
        text = re.sub(r'\d+','',text)

    for t in tokens:
        text = re.sub(r'\b' + re.escape(t) + r'\b', '', text)

    if no_stopwords:
        stop_words = set(stopwords.words('english')) 
        text = " ".join( [word for word in text.split() if word not in stop_words] )

    return text

def generate_reverse_tfidf(texts, method='sublinear'):
    vocab = set()
    doc_size = len(texts)
    tf = []
    idf = {}
    tf_idf = []
    tf_idf_array = []

    # Getting the vocab
    # ------------------

    for text in texts:
        text = clean_text(text)
        words = []
        for word in text.split():
            vocab.add(word)
            if word not in words:
                idf[word] = idf.get(word,0) + 1
            words.append(word)

    vocab = sorted(vocab)

    # Getting the TF
    # ------------------

    for text in texts:
        text = clean_text(text)
        tf_value = {}
        for word in vocab:
            tf_value[word] = 0
        size = len(text)
        _tf = Counter(text.split())
        for word ,value in _tf.items():
            tf_value[word] = value / size
        tf.append(tf_value)


    # Getting the IDF
    # ------------------

    for word in vocab:
        if method == 'smoothing':
            idf[word] = 1 + math.log( doc_size / (idf[word] +1 ) )
        elif method == 'maxidf':
            idf[word] = min( max(idf), math.log( doc_size / (idf[word] +1 ) ) )
        else:
            idf[word] = 1 + math.log( doc_size / idf[word] )


    # Combining the TF with the IDF
    # ------------------

    for i,t in enumerate(tf):
        result = {}
        for word,value in t.items():
            result[word] = value * idf[word]
        tf_idf.append(result)


    for i,t in enumerate(tf):
        result = []
        for word,value in t.items():
            result.append( value * idf[word] )
        tf_idf_array.append(result)



    return tf_idf_array


def generate_tfidf(texts, method='sublinear'):
    vocab = set()
    doc_size = len(texts)
    tf = {}
    idf = {}
    tf_idf = {}
    tf_idf_array = []

    # Getting the vocab
    # ------------------

    for text in texts:
        text = clean_text(text)
        words = []
        for word in text.split():
            vocab.add(word)
            tf[word] = [0] * doc_size
            tf_idf[word] = [0] * doc_size
            if word not in words:
                idf[word] = idf.get(word,0) + 1
            words.append(word)

    vocab = sorted(vocab)
    tf = sorted(tf)
    idf = sorted(idf)

    # Getting the TF
    # ------------------

    for i,text in enumerate(texts):
        text = clean_text(text)
        size = len(text)
        _tf = Counter(text.split())
        for word ,value in _tf.items():
            tf[word][i] = (tf[word][i] + 1) / size


    # Getting the IDF
    # ------------------

    for word in vocab:
        if method == 'smoothing':
            idf[word] = 1 + math.log( doc_size / (idf[word] +1 ) )
        elif method == 'maxidf':
            idf[word] = min( max(idf), math.log( doc_size / (idf[word] +1 ) ) )
        else:
            idf[word] = 1 + math.log( doc_size / idf[word] )


    # Combining the TF with the IDF
    # ------------------

    for i,text in enumerate(texts):
        for word ,value in tf.items():
            tf_idf[word][i] = tf[word][i] * idf[word]

    for key,value in tf_idf.items():
        tf_idf_array.append(value)
        


    return tf_idf_array


def generate_bow(texts):
    vocab = set()
    doc_size = len(texts)
    bow = {}
    bow_array = []

    # Getting the vocab
    # ------------------

    for text in texts:
        text = clean_text(text)
        for word in text.split():
            vocab.add(word)
            bow[word] = [0] * doc_size

    vocab = sorted(vocab)

    for i,text in enumerate(texts):
        text = clean_text(text)
        size = len(text)
        _tf = Counter(text.split())
        for word ,value in _tf.items():
            bow[word][i] = (bow[word][i] + 1) 

    for key,value in bow.items():
        bow_array.append(value)

    return bow_array


def generate_ngram(texts, n=2):
    vocab = set()
    doc_size = len(texts)

    ngram = {}

    # Getting the vocab
    # ------------------

    for text in texts:
        text = clean_text(text)
        words = text.split()
        for word in words:
            vocab.add(word)

        for i in range(len(words) - n + 1):
            
            context = tuple(words[i:i+n-1]) 
            next_word = words[i+n-1]
            if context not in ngram:
              ngram[context] = {}
            if next_word not in ngram[context]:
              ngram[context][next_word] = 0
            ngram[context][next_word] += 1


    return ngram

def one_hot_encode(data):
    encoded_data = []
    unique_values = list(set(data))
    for value in data:
        one_hot_vector = [0] * len(unique_values)
        index = unique_values.index(value)
        one_hot_vector[index] = 1
        encoded_data.append(one_hot_vector)

    return encoded_data


