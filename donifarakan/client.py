import pandas as pd
import numpy as np
from .utils import load_dataset, clean_text, generate_tfidf, generate_reverse_tfidf
import os
from  termcolor import colored
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import joblib
import pickle
import requests
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder
import re


agg_methods = {'1':"Federated Averaging (FedAvg)", '2':"Federated Matched Averaging (FedMA)", '3':"All Model Averaging (AMA)", '4': "One Model Selection (OMS)", '5':"Best Models Averaging (BMA)", '6': "FedProx", '7': "Hybrid Approaches"}
models = {'1': 'Linear Regression', '2': 'Logistic Regression', '3': 'Mutli-Layer Perceptron (MLP)', '4': 'Long-Short Term Memory (LSTM)'}
all_models = { '1': LinearRegression(), '2': LogisticRegression(random_state=16) ,'3': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500), '4': Sequential() }
regions = {'1': 'Africa', '2': 'America', '3': 'Middle east', '4': 'Europe', '5': 'Asia', '6': 'World Wide', '7': 'Non-classified'}
cats = {'1': 'Stock Prices', '2': 'News Sentiment', '3': 'Other (You will specify the features later)' }
markets = {'1': 'Technology Market', '2': 'Blue-Chip Market', '3': 'Emerging Markets', '4': 'Energy & Oil Market', '5': 'Financial Market (Banking and Insurance)', '6': 'Healthcare & Pharmaceutical Market', '7': 'Consumer Goods & Retail Market', '8':'Industrial & Manufacturing Market', '9':'Real Estate Market (REITs)', '10':'Telecommunications Market', '11':'Cryptocurrency & Blockchain Market'}
markets_examples = {'1': 'NASDAQ (USA)', '2': 'Dow Jones Industrial Average (DJIA) (USA), FTSE 100 (UK)', '3':'Nifty 50 (India), Shanghai Stock Exchange (China), Bovespa (Brazil)', '4':'S&P Global Energy Index, NYSE Arca Oil Index (XOI)', '5':'S&P Financials Index, KBW Bank Index (BKX)', '6':'NYSE Healthcare Index, NASDAQ Biotechnology Index', '7':'S&P Consumer Discretionary Index, NYSE Retail Index', '8':'Dow Jones Transportation Index, S&P Industrials Index', '9':'S&P Real Estate Index, FTSE NAREIT Equity REITs Index', '10':'S&P Communications Index, NYSE Telecom Index', '11':'NASDAQ Crypto Index, Coinbase Stock (COIN), Bitcoin ETFs'}
markets_details = {'1': 'Includes companies in software, hardware, semiconductors, cloud computing, AI, and cybersecurity.', '2': 'Composed of well-established, financially stable companies with a long track record.', '3': 'Includes stocks from developing countries with high growth potential.', '4': 'Focused on oil, gas, renewable energy, and utilities.', '5': 'Covers banking, asset management, fintech, and insurance companies.', '6': 'Includes biotech, pharmaceuticals, hospitals, and medical device companies.', '7': 'Includes luxury brands, fast-moving consumer goods (FMCG), and e-commerce.', '8':'Includes aerospace, defense, transportation, construction, and heavy machinery.', '9':'Composed of companies investing in real estate properties and development.', '10':'Covers internet providers, mobile network operators, and satellite communications.', '11':'Includes companies involved in crypto exchanges, blockchain technology, and DeFi.'}
markets_companies = {'1': 'Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Google (GOOGL), Amazon (AMZN),...', '2': 'Coca-Cola (KO), Johnson & Johnson (JNJ), IBM, McDonald\'s (MCD), Procter & Gamble (PG),...', '3':'Reliance Industries (India), Alibaba (China), Vale (Brazil), Tata Motors (India),...', '4':'ExxonMobil (XOM), Chevron (CVX), BP, Shell, Saudi Aramco,...', '5':'JPMorgan Chase (JPM), Goldman Sachs (GS), Wells Fargo (WFC), Visa (V), PayPal (PYPL),...', '6':'Pfizer (PFE), Moderna (MRNA), Johnson & Johnson (JNJ), Merck (MRK), Roche (ROG),...', '7':'Walmart (WMT), Amazon (AMZN), Nike (NKE), Procter & Gamble (PG), Tesla (TSLA),...', '8':'Boeing (BA), Caterpillar (CAT), Lockheed Martin (LMT), General Electric (GE),...', '9':'Simon Property Group (SPG), Prologis (PLD), Public Storage (PSA),...', '10':'AT&T (T), Verizon (VZ), T-Mobile (TMUS), Vodafone (VOD),...', '11':'Coinbase (COIN), MicroStrategy (MSTR), Bitcoin ETFs (BITO, IBIT), Riot Blockchain (RIOT),...'}

model_extension=".joblib"

#------------------------------------------
def convert_quoted_numbers(value):
    if isinstance(value, str):  # Check if the value is a string
        value = value.replace('"', '').replace(',', '')  # Remove quotes and commas
        try:
            return float(value)  # Convert to float
        except ValueError:
            return value  # Handle conversion error
    return value

#------------------------------------------
def send_model_to_server(server_url,model_filename,model_file,data,source_models_path):
    with open(model_filename, 'rb') as model_file:
        files = {'model': model_file}
        response = requests.post(server_url, files=files, data=data) #,stream=True)

        if response.status_code == 200 and response.headers.get('Content-Type') != 'application/json':
            extension = os.path.splitext(data['filename'])[1]
            with open(os.path.join(source_models_path,'global_model'+extension),'wb') as gl_model:
                for chunck in response.iter_content(chunk_size=8192):
                    gl_model.write(chunck)
                print(colored(f"\t|>> Global model updated successfully !\n", 'green'))
        else:
            # print(response.json())
            print(colored(f"\n|> Error while receiving the global model: {response.json()}",'red'))


#------------------------------------------
def test():
    try:
        print("------------------------------------------")
        print("TEST OF THE GLOBAL MODEL")
        print("------------------------------------------")

        print('|> Provide the full directory path that contains all the test datasets:')
        print('\n\t|>> ',end="")
        dataset_path = input("").strip()
        dataset_path = dataset_path if dataset_path != "" else "/home/donifaranga/datasets"
        print(colored(f"\t|>> {dataset_path}\n",'blue'))

        print('|> Please specify the market categorie on which the dataset is based on:')
        print('\n\t|>> ',end="")
        print('--')
        for index,name in markets.items():
            print('\t',index,". "+name, f" ( {markets_details[index]}) ")
            # print('\t\t> ',markets_details[index])
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

        print('|> Please specify the categorie of the dataset to test the Global model:')
        print('\n\t|>> ',end="")
        print('--')
        for index,name in cats.items():
            print('\t',index,". "+name)
        print('\n\t|>> ',end="")
        cat_index = input("").strip()
        cat_index = cat_index if cat_index in cats.keys() else "1"
        print(colored(f"\t|>> {cats[cat_index]}\n", 'blue'))

        cat = cats[cat_index]

        source_stock_price = os.path.join(dataset_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(), cats['1'].replace(" ", "-").lower())
        source_dataset = os.path.join(dataset_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(), cat.replace(" ", "-").lower())

        print('|> Select the model for training:')
        print('\n\t|>> ',end="")
        print('--')
        for index,name in models.items():
            print('\t',index,". "+name)
        print('\n\t|>> ',end="")
        model_index = input("").strip()
        model_index = model_index if model_index in models.keys() else "2"
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

        print('|> Specify the full directory path where you want to save the trained model:')
        print('\n\t|>> ',end="")
        models_path = input("").strip()
        models_path = models_path if models_path != "" else "/home/donifaranga/models"
        print(colored(f"\t|>> {models_path}\n",'blue'))

        source_models_path = os.path.join(models_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower(),models[model_index].replace(" ", "-").lower(),agg_methods[agg_index].replace(" ", "-").lower())



        if not os.path.exists(source_dataset):
            os.makedirs(source_dataset)

        if not os.path.exists(source_models_path):
            os.makedirs(source_models_path)

        print()

        dataframes = []

        for data_src in os.listdir(source_dataset):
            data_path = os.path.join(source_dataset,data_src)
            if data_src.startswith('.') == False and os.path.isfile(data_path):
                data_set = load_dataset(data_path)
                dataframes.append(data_set)

        dataset = pd.concat(dataframes, ignore_index=True)

        print('\n-----------------------------------| [TESTING]\n')

        # print('\n|> Checking if there are too many missing values:')
        # print(dataset.isnull().sum())
        dataset.dropna(inplace=True)
        # dataset.drop_duplicates(inplace=True)
        # dataset = dataset[(dataset[['Open', 'Close', 'High', 'Low', 'Volume']] >= 0).all(axis=1)]
        # dataset['Adj Close'] = dataset['Close'] * adjustment_factor

        # print('\n|> Filter out non-trading days:')
        # non_trading_days = dataset[dataset['Volume'] == 0]
        # print(non_trading_days)



        if cat_index == '1':
            features_columns = ['Open', 'High', 'Low', 'Volume']
            target_column = 'Close'

            dataset = dataset.apply(lambda col: col.map(convert_quoted_numbers)) #dataset.applymap(convert_quoted_numbers)

            X = dataset[features_columns] #.drop(columns=[target_column])
            y = dataset[target_column]

            # dataset.drop_duplicates(inplace=True)
            # dataset = dataset[(dataset[['Open', 'Close', 'High', 'Low', 'Volume']] >= 0).all(axis=1)]
            # dataset['Adj Close'] = dataset['Close'] * adjustment_factor

        elif cat_index == '2':
            features_columns = ['News','Close']
            target_column = 'Direction_Binary' 


            dataset = dataset.groupby('Date')['News'].agg(lambda x: ' '.join(x)).reset_index()
            dataset = dataset.drop_duplicates(subset=['Date'], keep=False)

            stock_dataframes = []

            print("|> Loading stock prices datasets...")
            for data_src in os.listdir(source_stock_price):
                data_path = os.path.join(source_stock_price,data_src)
                if data_src.startswith('.') == False and os.path.isfile(data_path):
                    data_set = load_dataset(data_path)
                    stock_dataframes.append(data_set)

            print("|> Concatenating...")
            stock_dataset = pd.concat(stock_dataframes, ignore_index=True)
            stock_dataset = stock_dataset.dropna(how='any')

            print("|> Date conversion...")
            stock_dataset['Date'] = pd.to_datetime(stock_dataset['Date'])
            dataset['Date'] = pd.to_datetime(dataset['Date'])

            print("|> Merging stock data with news data...")
            dataset = pd.merge(dataset, stock_dataset, on='Date', how='left')
            dataset = dataset.drop_duplicates(subset=['Date'], keep=False)

            dataset['News'] = dataset['News'].fillna('No news available')

            print("|> Getting Previous and Next close...")
            dataset['Previous_Close'] = dataset['Close'].shift(1).fillna(dataset['Close'].iloc[0])  # Previous day's close

            print("|> Getting movement...")
            dataset['Movement'] = dataset['Close'] - dataset['Previous_Close']
            dataset = dataset.drop('Previous_Close', axis=1)

            print("|> Setting Direction values...")
            dataset['Direction'] = dataset['Movement'].apply(lambda x: 'Up' if x >= 0 else 'Down')
            dataset['Direction_Binary'] = dataset['Direction'].map({'Up': 1, 'Down': 0})


            dataset = dataset.dropna(how='any')
            dataset = dataset.drop_duplicates()
            print("|> Converting News text into TF-IDF vectors...")
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            X = tfidf_vectorizer.fit_transform(dataset['News'])

            # X = generate_reverse_tfidf(dataset['News']) #.drop(columns=[target_column])
            y = dataset[target_column]
            
        else:
            print('|> Please provide the list of columns to consider (separated by comma and as they appear in the dataset):')
            print('\n\t|>> ',end="")
            features_columns = input("")
            features_columns = [ item.strip() for item in features_columns.split(',') ]
            print(colored(f"\t|>>  {features_columns}\n",'blue'))

            print('|> Please specify the target column name (as it appears in the dataset):')
            print('\n\t|>> ',end="")
            target_column = input("")
            print(colored(f"\t|>>  {target_column}\n",'blue'))

            if dataset[target_column].dtype == 'object':
                label_encoder = LabelEncoder()
                dataset[target_column] = label_encoder.fit_transform(dataset[target_column])

            X = dataset[features_columns] #.drop(columns=[target_column])
            y = dataset[target_column]


        print(dataset)
        print(f"\n|> Dataset shape: {dataset.shape} ")

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # X_scaled = scaler.fit_transform(X)
        # y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

        # scaler_filename = os.path.join(source_models_path, 'scaler.joblib')
        # joblib.dump(scaler, scaler_filename)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # -----------------
        # REGRESSION
        # -----------------
        if model_index == '1':
            print("\n|> Testing on: [ Regression model ]")

            model_path = os.path.join(source_models_path, 'global_model'+model_extension)

            if os.path.exists(model_path):
                if model_path.endswith('.joblib'):
                    model = joblib.load(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as nm_file:
                        model = pickle.load(nm_file)
                elif model_path.endswith('.keras'):
                    model = load_model(model_path)
                else:
                    model = None
            else:
                model = all_models[model_index] #LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mean_actual = np.mean(y_test)
            accuracy = r2_score(y_test, y_pred)
            error = mse


            # print("|>> SHAP summary")
            # background = shap.maskers.Independent(X_train, max_samples=100)
            # explainer = shap.Explainer(model.predict,background)
            # shap_values = explainer(X_test)
            # shap.summary_plot(shap_values, X_test)

            # result = permutation_importance(model, X_test, y_test, n_repeats=10)
            # importance = result.importances_mean
            # importance_df = pd.DataFrame({
            # 'Feature': features_columns,
            # 'Importance': importance,
            # 'Std': result.importances_std
            # })
            # importance_df = importance_df.sort_values(by='Importance', ascending=False)
            # plt.figure(figsize=(12, 6))
            # plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], color='skyblue')
            # plt.xlabel('Importance (Performance drop)')
            # plt.title('Permutation Feature Importance')
            # plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
            # plt.show()

        # -----------------
        # LOGISTIC REGRESSION
        # -----------------
        elif model_index == '2':
            print("\n|> Testing on: [ Logistic Regression model ]")

            model_path = os.path.join(source_models_path, 'global_model'+model_extension)

            if os.path.exists(model_path):
                if model_path.endswith('.joblib'):
                    regressor = joblib.load(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as nm_file:
                        regressor = pickle.load(nm_file)
                elif model_path.endswith('.keras'):
                    regressor = load_model(model_path)
                else:
                    regressor = None
            else:
                regressor = all_models[model_index] #LinearRegression()
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mean_actual = np.mean(y_test)
            accuracy = accuracy_score(y_test, y_pred)
            error = mse

            # print("|>> SHAP summary")
            # background = shap.maskers.Independent(X_train, max_samples=100)
            # explainer = shap.Explainer(regressor.predict,background)
            # shap_values = explainer(X_test)
            # shap.summary_plot(shap_values, X_test)

            # result = permutation_importance(regressor, X_test, y_test, n_repeats=10)
            # importance = result.importances_mean
            # importance_df = pd.DataFrame({
            # 'Feature': features_columns,
            # 'Importance': importance,
            # 'Std': result.importances_std
            # })
            # importance_df = importance_df.sort_values(by='Importance', ascending=False)
            # plt.figure(figsize=(12, 6))
            # plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], color='skyblue')
            # plt.xlabel('Importance (Performance drop)')
            # plt.title('Permutation Feature Importance')
            # plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
            # plt.show()


        # -----------------
        # MLP
        # -----------------
        elif model_index == '3':
            print("\n|> Testing: [ MLP model ]")

            model_path = os.path.join(source_models_path, 'global_model'+model_extension)
            
            if os.path.exists(model_path):
                if model_path.endswith('.joblib'):
                    model = joblib.load(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as nm_file:
                        model = pickle.load(nm_file)
                elif model_path.endswith('.keras'):
                    model = load_model(model_path)
                else:
                    model = None
            else:
                model = all_models[model_index]

            model = all_models[model_index] #MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Evaluate performance
            mse_mlp = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse_mlp)
            mean_actual = np.mean(y_test)
            accuracy = r2_score(y_test, y_pred)
            error = mse_mlp


        # -----------------
        # LSTM
        # -----------------
        elif model_index == '4':
            print("\n|> Testing on: [ LSTM model ]")

            model_path = os.path.join(source_models_path, 'global_model.keras')
            
            if os.path.exists(model_path):
                model = load_model(model_path)
            else:

                X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                model = Sequential()

                # Add an LSTM layer with 50 units and a Dense output layer
                model.add(LSTM(units=50, return_sequences=False, input_shape=(1, X_train.shape[1])))
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')


            # Train the model
            model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)

            # Predict on the test set
            y_pred = model.predict(X_test_lstm)

            # Evaluate performance
            mse_lstm = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse_lstm)
            mean_actual = np.mean(y_test)
            accuracy = r2_score(y_test, y_pred)
            error = mse_lstm


        print()
        print(colored(f'\t|>> Predicted target values: [ Accuracy: {accuracy * 100:.2f}% ]', 'blue'))
        print(colored(f'\t|----------------------------\n', 'blue'))
        print(colored(y_pred,'green'))


    except Exception as e:
        print(colored(f"|> Error occured during the process: {e}",'red'))

    print('\n---------------[END]\n')

#------------------------------------------
def global_model():
    try:
        print("------------------------------------------")
        print(" GETTING THE GLOBAL MODEL")
        print("------------------------------------------")


        print('|> Before the traning please specify the central server full access url:')
        print('\n\t|>> ',end="")
        server_url = input("").strip().replace(' ', '')
        server_url = server_url if server_url != "" else "http://localhost:5000/api/get_model"
        print(colored(f"\t|>> {server_url}\n",'blue'))

        print('|> Specify the full path where you want to save the global model:')
        print('\n\t|>> ',end="")
        models_path = input("").strip()
        models_path = models_path if models_path != "" else "/home/donifaranga/models"
        print(colored(f"\t|>> {models_path}\n",'blue'))

        print('|> Please specify the market categorie on which the dataset is based on:')
        print('\n\t|>> ',end="")
        print('--')
        for index,name in markets.items():
            print('\t',index,". "+name, f" ( {markets_details[index]}) ")
            # print('\t\t> ',markets_details[index])
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

        print('|> Please specify the type or categorie of the global model to use (stock price data, news data,...):')
        print('\n\t|>> ',end="")
        print('--')
        for index,name in cats.items():
            print('\t',index,". "+name)
        print('\n\t|>> ',end="")
        cat_index = input("").strip()
        cat = cats[cat_index] if cat_index in cats.keys() else "1"
        print(colored(f"\t|>> {cat}\n", 'blue'))

        print('|> Select the model for training:')
        print('\n\t|>> ',end="")
        print('--')
        for index,name in models.items():
            print('\t',index,". "+name)
        print('\n\t|>> ',end="")
        model_index = input("").strip()
        model_index = model_index if model_index in models.keys() else "2"
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


        source_models_path = os.path.join(models_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower(),models[model_index].replace(" ", "-").lower(),agg_methods[agg_index].replace(" ", "-").lower())

        if not os.path.exists(source_models_path):
            os.makedirs(source_models_path)

        data = {'cat': cat_index, 'model_index': model_index, 'region_index': region_index, 'market_index': market_index, 'agg_index': agg_index}
        response = requests.post(server_url, data=data) #,stream=True)

        if response.status_code == 200 and response.headers.get('Content-Type') != 'application/json':

            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition:
                # Use regex to find the filename
                filename = re.findall('filename="(.+?)"', content_disposition)
                if filename:
                    filename = filename[0]  # Get the first match
                else:
                    filename = 'global_model.joblib'  # Fallback if no filename found
            else:
                filename = 'global_model.joblib'  # Fallback if no Content-Disposition header

            with open(os.path.join(source_models_path,filename),'wb') as gl_model:
                for chunck in response.iter_content(chunk_size=8192):
                    gl_model.write(chunck)
                print(colored(f"\t|>> Global model downloaded successfully !\n", 'green'))
        else:
            print(colored(f"|> Error! occured during the process: {response.json()}",'red'))

    except Exception as e:
        print(colored(f"|> Error occured during the process: {e}",'red'))

    print('\n---------------[END]\n')

#------------------------------------------
def train():
    

    print("-------------------------------------------------------------")
    print(" TRAINING (SUPERVISED LEARNING)")
    print("-------------------------------------------------------------")

    print('|> Before the traning please specify the central server full access url:')
    print('\n\t|>> ',end="")
    server_url = input("").strip().replace(' ', '')
    server_url = server_url if server_url != "" else "http://localhost:5000/api/receive_model"
    print(colored(f"\t|>>  {server_url}\n",'blue'))

    print('|> Your client ID:')
    print('\n\t|>> ',end="")
    client_id = input("").strip()
    client_id = 'Client-'+client_id if client_id != "" else "001"
    print(colored(f"\t|>>  {client_id}\n",'blue'))

    print('|> Provide the full directory path that contains all the datasets:')
    print('\n\t|>> ',end="")
    dataset_path = input("").strip()
    dataset_path = dataset_path if dataset_path != "" else "/home/donifaranga/datasets"
    print(colored(f"\t|>> {dataset_path}\n",'blue'))

    print('|> Please specify the market categorie on which the dataset is based on:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in markets.items():
        print('\t',index,". "+name)
        print('\t\t> ',markets_details[index])
        print('\t\t> ',markets_companies[index])
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


    print('|> Please specify the type or categorie of the dataset to use (stock price data, news data,...):')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in cats.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    cat_index = input("").strip()
    cat = cats[cat_index] if cat_index in cats.keys() else "1"
    print(colored(f"\t|>> {cat}\n", 'blue'))

    source_dataset = os.path.join(dataset_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower())
    source_stock_price = os.path.join(dataset_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cats['1'].replace(" ", "-").lower())

    if not os.path.exists(source_dataset):
        os.makedirs(source_dataset)

    print('|> Select the model for training:')
    print('\n\t|>> ',end="")
    print('--')
    for index,name in models.items():
        print('\t',index,". "+name)
    print('\n\t|>> ',end="")
    model_index = input("").strip()
    model_index = model_index if model_index in models.keys() else "2"
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

    print('|> Specify the full directory path where you want to save the trained model:')
    print('\n\t|>> ',end="")
    models_path = input("").strip()
    models_path = models_path if models_path != "" else "/home/donifaranga/models"
    print(colored(f"\t|>> {models_path}\n",'blue'))

    source_models_path = os.path.join(models_path,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower(),models[model_index].replace(" ", "-").lower(),agg_methods[agg_index].replace(" ", "-").lower())

    if not os.path.exists(source_models_path):
        os.makedirs(source_models_path)

    print()


    try:

        dataframes = []

        for data_src in os.listdir(source_dataset):
            data_path = os.path.join(source_dataset,data_src)
            if data_src.startswith('.') == False and os.path.isfile(data_path):
                data_set = load_dataset(data_path)
                dataframes.append(data_set)

        dataset = pd.concat(dataframes, ignore_index=True)

        print('\n-----------------------------------| [TRAINING]\n')

        # print('\n|> Checking if there are too many missing values:')
        # print(dataset.isnull().sum())
        dataset.dropna(inplace=True)
        

        # print('\n|> Filter out non-trading days:')
        # non_trading_days = dataset[dataset['Volume'] == 0]
        # print(non_trading_days)

        if cat_index == '1':
            features_columns = ['Open', 'High', 'Low', 'Volume']
            target_column = 'Close'

            dataset = dataset.apply(lambda col: col.map(convert_quoted_numbers)) #dataset.applymap(convert_quoted_numbers)

            X = dataset[features_columns] #.drop(columns=[target_column])
            y = dataset[target_column]

            # dataset.drop_duplicates(inplace=True)
            # dataset = dataset[(dataset[['Open', 'Close', 'High', 'Low', 'Volume']] >= 0).all(axis=1)]
            # dataset['Adj Close'] = dataset['Close'] * adjustment_factor

        elif cat_index == '2':
            features_columns = ['News','Close']
            target_column = 'Direction_Binary' 


            dataset = dataset.groupby('Date')['News'].agg(lambda x: ' '.join(x)).reset_index()
            dataset = dataset.drop_duplicates(subset=['Date'], keep=False)

            stock_dataframes = []

            print("|> Loading stock prices datasets...")
            for data_src in os.listdir(source_stock_price):
                data_path = os.path.join(source_stock_price,data_src)
                if data_src.startswith('.') == False and os.path.isfile(data_path):
                    data_set = load_dataset(data_path)
                    stock_dataframes.append(data_set)

            print("|> Concatenating...")
            stock_dataset = pd.concat(stock_dataframes, ignore_index=True)
            stock_dataset = stock_dataset.dropna(how='any')

            print("|> Date conversion...")
            stock_dataset['Date'] = pd.to_datetime(stock_dataset['Date'])
            dataset['Date'] = pd.to_datetime(dataset['Date'])

            print("|> Merging stock data with news data...")
            dataset = pd.merge(dataset, stock_dataset, on='Date', how='left')
            dataset = dataset.drop_duplicates(subset=['Date'], keep=False)

            dataset['News'] = dataset['News'].fillna('No news available')

            print("|> Getting Previous and Next close...")
            dataset['Previous_Close'] = dataset['Close'].shift(1).fillna(dataset['Close'].iloc[0])  # Previous day's close

            print("|> Getting movement...")
            dataset['Movement'] = dataset['Close'] - dataset['Previous_Close']
            dataset = dataset.drop('Previous_Close', axis=1)

            print("|> Setting Direction values...")
            dataset['Direction'] = dataset['Movement'].apply(lambda x: 'Up' if x >= 0 else 'Down')
            dataset['Direction_Binary'] = dataset['Direction'].map({'Up': 1, 'Down': 0})


            dataset = dataset.dropna(how='any')
            dataset = dataset.drop_duplicates()
            print("|> Converting News text into TF-IDF vectors...")
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            X = tfidf_vectorizer.fit_transform(dataset['News'])

            # X = generate_reverse_tfidf(dataset['News']) #.drop(columns=[target_column])
            y = dataset[target_column]
            
        else:
            print('|> Please provide the list of columns to consider (separated by comma and as they appear in the dataset):')
            print('\n\t|>> ',end="")
            features_columns = input("")
            features_columns = [ item.strip() for item in features_columns.split(',') ]
            print(colored(f"\t|>>  {features_columns}\n",'blue'))

            print('|> Please specify the target column name (as it appears in the dataset):')
            print('\n\t|>> ',end="")
            target_column = input("")
            print(colored(f"\t|>>  {target_column}\n",'blue'))

            if dataset[target_column].dtype == 'object':
                label_encoder = LabelEncoder()
                dataset[target_column] = label_encoder.fit_transform(dataset[target_column])

            X = dataset[features_columns] #.drop(columns=[target_column])
            y = dataset[target_column]


        print(dataset)
        
        print(f"\n|> Dataset shape: {dataset.shape} ")

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # X_scaled = scaler.fit_transform(X)
        # y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

        # scaler_filename = os.path.join(source_models_path, 'scaler.joblib')
        # joblib.dump(scaler, scaler_filename)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # -----------------
        # LINEAR REGRESSION
        # -----------------
        if model_index == '1':
            print("\n|> Training on: [ Linear Regression model ]")

            model_path = os.path.join(source_models_path, 'global_model'+model_extension)

            if os.path.exists(model_path):
                if model_path.endswith('.joblib'):
                    regressor = joblib.load(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as nm_file:
                        regressor = pickle.load(nm_file)
                elif model_path.endswith('.keras'):
                    regressor = load_model(model_path)
                else:
                    regressor = None
            else:
                regressor = all_models[model_index] #LinearRegression()
            
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mean_actual = np.mean(y_test)
            accuracy = r2_score(y_test, y_pred)
            error = mse
            print(colored(f'\t|>> Linear Regression (accuracy): {accuracy * 100:.2f}%','blue'))


            lr_model_filename = os.path.join(source_models_path,'global_model'+model_extension)

            filename = os.path.basename(lr_model_filename)

            data = {'cat': cat_index, 'id': client_id, 'filename': filename,'agg':agg_index,'model_type':model_index, 'region_index': region_index, 'accuracy':accuracy, 'error': error,'market_index': market_index}

            if lr_model_filename.endswith('.pkl'):
                with open(lr_model_filename, 'wb') as file:
                    pickle.dump(regressor, file)
                    send_model_to_server(server_url,lr_model_filename,regressor,data,source_models_path)
            elif lr_model_filename.endswith('.joblib'):
                joblib.dump(regressor, lr_model_filename)
                send_model_to_server(server_url,lr_model_filename,regressor,data,source_models_path)
            else:
                print(colored(f"|> Error: Failed to save the trained model",'red'))

            # print("|>> SHAP summary")
            # background = shap.maskers.Independent(X_train, max_samples=100)
            # explainer = shap.Explainer(regressor.predict,background)
            # shap_values = explainer(X_test)
            # shap.summary_plot(shap_values, X_test)

            # result = permutation_importance(regressor, X_test, y_test, n_repeats=10)
            # importance = result.importances_mean
            # importance_df = pd.DataFrame({
            # 'Feature': features_columns,
            # 'Importance': importance,
            # 'Std': result.importances_std
            # })
            # importance_df = importance_df.sort_values(by='Importance', ascending=False)
            # plt.figure(figsize=(12, 6))
            # plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], color='skyblue')
            # plt.xlabel('Importance (Performance drop)')
            # plt.title('Permutation Feature Importance')
            # plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
            # plt.show()

        # -----------------
        # LOGISTIC REGRESSION
        # -----------------
        elif model_index == '2':
            print("\n|> Training on: [ Logistic Regression model ]")

            model_path = os.path.join(source_models_path, 'global_model'+model_extension)

            if os.path.exists(model_path):
                if model_path.endswith('.joblib'):
                    regressor = joblib.load(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as nm_file:
                        regressor = pickle.load(nm_file)
                elif model_path.endswith('.keras'):
                    regressor = load_model(model_path)
                else:
                    regressor = None
            else:
                regressor = all_models[model_index] #LinearRegression()
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mean_actual = np.mean(y_test)
            accuracy = accuracy_score(y_test, y_pred)
            error = mse
            print(colored(f'\t|>> Logistic Regression (accuracy): {accuracy * 100:.2f}%','blue'))


            lr_model_filename = os.path.join(source_models_path,'global_model'+model_extension)

            filename = os.path.basename(lr_model_filename)

            data = {'cat': cat_index, 'id': client_id, 'filename': filename,'agg':agg_index,'model_type':model_index, 'region_index': region_index, 'accuracy':accuracy, 'error': error, 'market_index': market_index}

            if lr_model_filename.endswith('.pkl'):
                with open(lr_model_filename, 'wb') as file:
                    pickle.dump(regressor, file)
                    send_model_to_server(server_url,lr_model_filename,regressor,data,source_models_path)
            elif lr_model_filename.endswith('.joblib'):
                joblib.dump(regressor, lr_model_filename)
                send_model_to_server(server_url,lr_model_filename,regressor,data,source_models_path)
            else:
                print(colored(f"|> Error: Failed to save the trained model",'red'))

            print("|>> SHAP summary")
            background = shap.maskers.Independent(X_train, max_samples=100)
            explainer = shap.Explainer(regressor.predict,background)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test)

            # result = permutation_importance(regressor, X_test, y_test, n_repeats=10)
            # importance = result.importances_mean
            # importance_df = pd.DataFrame({
            # 'Feature': features_columns,
            # 'Importance': importance,
            # 'Std': result.importances_std
            # })
            # importance_df = importance_df.sort_values(by='Importance', ascending=False)
            # plt.figure(figsize=(12, 6))
            # plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], color='skyblue')
            # plt.xlabel('Importance (Performance drop)')
            # plt.title('Permutation Feature Importance')
            # plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
            # plt.show()

        # -----------------
        # MLP
        # -----------------
        elif model_index == '3':
            print("\n|> Training: [ MLP model ]")

            model_path = os.path.join(source_models_path, 'global_model'+model_extension)
            
            if os.path.exists(model_path):
                if model_path.endswith('.joblib'):
                    mlp = joblib.load(model_path)
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as nm_file:
                        mlp = pickle.load(nm_file)
                elif model_path.endswith('.keras'):
                    mlp = load_model(model_path)
                else:
                    mlp = None
            else:
                mlp = all_models[model_index]

            mlp = all_models[model_index] #MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
            mlp.fit(X_train, y_train)

            # Predict on the test set
            y_pred_mlp = mlp.predict(X_test)

            # Evaluate performance
            mse_mlp = mean_squared_error(y_test, y_pred_mlp)
            rmse = np.sqrt(mse_mlp)
            mean_actual = np.mean(y_test)
            accuracy = r2_score(y_test, y_pred_mlp)
            error = mse_mlp
            print(colored(f'\t|>> Neural Network MLP (accuracy): {accuracy * 100:.2f}%','blue'))

            mlp_model_filename = os.path.join(source_models_path,'global_model'+model_extension)

            filename = os.path.basename(mlp_model_filename)

            data = {'cat': cat_index, 'id': client_id, 'filename': filename,'agg':agg_index,'model_type':model_index, 'region_index': region_index,'accuracy':accuracy, 'error': error}

            if mlp_model_filename.endswith('.pkl'):
                with open(mlp_model_filename, 'wb') as file:
                    pickle.dump(mlp, file)
                    send_model_to_server(server_url,mlp_model_filename,mlp,data,source_models_path)
            elif mlp_model_filename.endswith('.joblib'):
                joblib.dump(mlp, mlp_model_filename)
                send_model_to_server(server_url,mlp_model_filename,mlp,data,source_models_path)
            else:
                print(colored(f"|> Error: Failed to save the trained model",'red'))



        # -----------------
        # LSTM
        # -----------------
        elif model_index == '4':
            print("\n|> Training: [ LSTM model ]")

            model_path = os.path.join(source_models_path, 'global_model.keras')
            
            if os.path.exists(model_path):
                model = load_model(model_path)
            else:

                X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                model = Sequential()

                # Add an LSTM layer with 50 units and a Dense output layer
                model.add(LSTM(units=50, return_sequences=False, input_shape=(1, X_train.shape[1])))
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')


            # Train the model
            model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)

            # Predict on the test set
            y_pred_lstm = model.predict(X_test_lstm)

            # Evaluate performance
            mse_lstm = mean_squared_error(y_test, y_pred_lstm)
            rmse = np.sqrt(mse_lstm)
            mean_actual = np.mean(y_test)
            accuracy = r2_score(y_test, y_pred_lstm)
            error = mse_lstm
            print(f'\tLSTM (accuracy): {accuracy * 100:.2f}%')

            lstm_model_filename = os.path.join(source_models_path,'global_model.keras')

            model.save(lstm_model_filename)

            filename = lstm_model_filename

            data = {'cat': cat_index, 'id': client_id, 'filename': filename,'agg':agg_index,'model_type':model_index, 'region_index': region_index,'accuracy':accuracy, 'error': error,'market_index': market_index}


            send_model_to_server(server_url,lstm_model_filename,mlp,data,source_models_path)


        # dataset['Close'].plot()
        # plt.show()

    except Exception as e:
        print(colored(f"|> Error occured during the process: {e}",'red'))

    print('\n---------------[END]\n')




