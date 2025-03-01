from flask import Blueprint, jsonify, request, send_file
from ..config import *
import pandas as pd
import numpy as np
from ..utils import load_dataset
import os
from  termcolor import colored
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import joblib
import pickle
import requests
import re


#------------------------------------------
# INITIALISATION
#------------------------------------------

api_bp = Blueprint('api', __name__)

@api_bp.route('/api', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

@api_bp.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, World!"})

@api_bp.route('/api/get_model', methods=['POST'])
def get_model():
    try:
        source_dir = os.path.dirname(os.path.dirname(__file__))
        source_dataset = os.path.join(source_dir,'datasets')
        source_models = os.path.join(source_dir,'models')

        agg_methods = {'1':"Federated Averaging (FedAvg)", '2':"Federated Matched Averaging (FedMA)", '3':"All Model Averaging (AMA)", '4': "One Model Selection (OMS)", '5':"Best Models Averaging (BMA)", '6': "FedProx", '7': "Hybrid Approaches"}
        all_models_name = {'1': 'Linear Regression', '2': 'Logistic Regression', '3': 'Mutli-Layer Perceptron (MLP)', '4': 'Long-Short Term Memory (LSTM)'}
        all_models = { '1': LinearRegression(), '2': LogisticRegression(random_state=16) ,'3': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500), '4': Sequential() }
        regions = {'1': 'Africa', '2': 'America', '3': 'Middle east', '4': 'Europe', '5': 'Asia', '6': 'World Wide', '7': 'Non-classified'}
        cats = {'1': 'Stock Prices', '2': 'News Sentiment', '3': 'Other (You will specify the features later)' }
        markets = {'1': 'Technology Market', '2': 'Blue-Chip Market', '3': 'Emerging Markets', '4': 'Energy & Oil Market', '5': 'Financial Market (Banking and Insurance)', '6': 'Healthcare & Pharmaceutical Market', '7': 'Consumer Goods & Retail Market', '8':'Industrial & Manufacturing Market', '9':'Real Estate Market (REITs)', '10':'Telecommunications Market', '11':'Cryptocurrency & Blockchain Market'}
        markets_examples = {'1': 'NASDAQ (USA)', '2': 'Dow Jones Industrial Average (DJIA) (USA), FTSE 100 (UK)', '3':'Nifty 50 (India), Shanghai Stock Exchange (China), Bovespa (Brazil)', '4':'S&P Global Energy Index, NYSE Arca Oil Index (XOI)', '5':'S&P Financials Index, KBW Bank Index (BKX)', '6':'NYSE Healthcare Index, NASDAQ Biotechnology Index', '7':'S&P Consumer Discretionary Index, NYSE Retail Index', '8':'Dow Jones Transportation Index, S&P Industrials Index', '9':'S&P Real Estate Index, FTSE NAREIT Equity REITs Index', '10':'S&P Communications Index, NYSE Telecom Index', '11':'NASDAQ Crypto Index, Coinbase Stock (COIN), Bitcoin ETFs'}
        markets_details = {'1': 'Includes companies in software, hardware, semiconductors, cloud computing, AI, and cybersecurity.', '2': 'Composed of well-established, financially stable companies with a long track record.', '3': 'Includes stocks from developing countries with high growth potential.', '4': 'Focused on oil, gas, renewable energy, and utilities.', '5': 'Covers banking, asset management, fintech, and insurance companies.', '6': 'Includes biotech, pharmaceuticals, hospitals, and medical device companies.', '7': 'Includes luxury brands, fast-moving consumer goods (FMCG), and e-commerce.', '8':'Includes aerospace, defense, transportation, construction, and heavy machinery.', '9':'Composed of companies investing in real estate properties and development.', '10':'Covers internet providers, mobile network operators, and satellite communications.', '11':'Includes companies involved in crypto exchanges, blockchain technology, and DeFi.'}
        markets_companies =  {'1': 'Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Google (GOOGL), Amazon (AMZN),...', '2': 'Coca-Cola (KO), Johnson & Johnson (JNJ), IBM, McDonald\'s (MCD), Procter & Gamble (PG),...', '3':'Reliance Industries (India), Alibaba (China), Vale (Brazil), Tata Motors (India),...', '4':'ExxonMobil (XOM), Chevron (CVX), BP, Shell, Saudi Aramco,...', '5':'JPMorgan Chase (JPM), Goldman Sachs (GS), Wells Fargo (WFC), Visa (V), PayPal (PYPL),...', '6':'Pfizer (PFE), Moderna (MRNA), Johnson & Johnson (JNJ), Merck (MRK), Roche (ROG),...', '7':'Walmart (WMT), Amazon (AMZN), Nike (NKE), Procter & Gamble (PG), Tesla (TSLA).', '8':'Boeing (BA), Caterpillar (CAT), Lockheed Martin (LMT), General Electric (GE),...', '9':'Simon Property Group (SPG), Prologis (PLD), Public Storage (PSA),...', '10':'AT&T (T), Verizon (VZ), T-Mobile (TMUS), Vodafone (VOD),...', '11':'Coinbase (COIN), MicroStrategy (MSTR), Bitcoin ETFs (BITO, IBIT), Riot Blockchain (RIOT),...'}

        cat_index = request.form.get('cat')
        region_index = request.form.get('region_index')
        agg_index = request.form.get('agg_index')
        model_index = request.form.get('model_index')
        market_index = request.form.get('market_index')

        source_models = os.path.join(source_models,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cats[cat_index].replace(" ", "-").lower(),all_models_name[model_index].replace(" ", "-").lower(),agg_methods[agg_index].replace(" ", "-").lower())
        print(source_models,'\n')

        if not os.path.exists(source_models):
            os.makedirs(source_models)


        model_name1 = os.path.join(source_models, 'global_model.joblib')
        model_name2 = os.path.join(source_models, 'global_model.pkl')
        model_name3 = os.path.join(source_models, 'global_model.keras')

        if os.path.exists(model_name1):
            print("Global model found!")
            return send_file(model_name1,as_attachment=True)
        elif os.path.exists(model_name2):
            print("Global model found!")
            return send_file(model_name2,as_attachment=True)
        elif os.path.exists(model_name3) :
            print("Global model found!")
            return send_file(model_name3,as_attachment=True)
        else:
            return jsonify({"error": "No global model file found"}), 404


    except Exception as e:
        return jsonify({"error": "Internal Server Error", "error_message": str(e)}), 500

@api_bp.route('/api/receive_model', methods=['POST'])
def receive_parameters():
    try:

        source_dir = os.path.dirname(os.path.dirname(__file__))
        source_dataset = os.path.join(source_dir,'datasets')
        source_models = os.path.join(source_dir,'models')
        clients_models = os.path.join(source_dir,'clients')

        agg_methods = {'1':"Federated Averaging (FedAvg)", '2':"Federated Matched Averaging (FedMA)", '3':"All Model Averaging (AMA)", '4': "One Model Selection (OMS)", '5':"Best Models Averaging (BMA)", '6': "FedProx", '7': "Hybrid Approaches"}
        all_models_name = {'1': 'Linear Regression', '2': 'Logistic Regression', '3': 'Mutli-Layer Perceptron (MLP)', '4': 'Long-Short Term Memory (LSTM)'}
        all_models = { '1': LinearRegression(), '2': LogisticRegression(random_state=16) ,'3': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500), '4': Sequential() }
        regions = {'1': 'Africa', '2': 'America', '3': 'Middle east', '4': 'Europe', '5': 'Asia', '6': 'World Wide', '7': 'Non-classified'}
        cats = {'1': 'Stock Prices', '2': 'News Sentiment', '3': 'Other (You will specify the features later)' }
        markets = {'1': 'Technology Market', '2': 'Blue-Chip Market', '3': 'Emerging Markets', '4': 'Energy & Oil Market', '5': 'Financial Market (Banking and Insurance)', '6': 'Healthcare & Pharmaceutical Market', '7': 'Consumer Goods & Retail Market', '8':'Industrial & Manufacturing Market', '9':'Real Estate Market (REITs)', '10':'Telecommunications Market', '11':'Cryptocurrency & Blockchain Market'}
        markets_examples = {'1': 'NASDAQ (USA)', '2': 'Dow Jones Industrial Average (DJIA) (USA), FTSE 100 (UK)', '3':'Nifty 50 (India), Shanghai Stock Exchange (China), Bovespa (Brazil)', '4':'S&P Global Energy Index, NYSE Arca Oil Index (XOI)', '5':'S&P Financials Index, KBW Bank Index (BKX)', '6':'NYSE Healthcare Index, NASDAQ Biotechnology Index', '7':'S&P Consumer Discretionary Index, NYSE Retail Index', '8':'Dow Jones Transportation Index, S&P Industrials Index', '9':'S&P Real Estate Index, FTSE NAREIT Equity REITs Index', '10':'S&P Communications Index, NYSE Telecom Index', '11':'NASDAQ Crypto Index, Coinbase Stock (COIN), Bitcoin ETFs'}
        markets_details = {'1': 'Includes companies in software, hardware, semiconductors, cloud computing, AI, and cybersecurity.', '2': 'Composed of well-established, financially stable companies with a long track record.', '3': 'Includes stocks from developing countries with high growth potential.', '4': 'Focused on oil, gas, renewable energy, and utilities.', '5': 'Covers banking, asset management, fintech, and insurance companies.', '6': 'Includes biotech, pharmaceuticals, hospitals, and medical device companies.', '7': 'Includes luxury brands, fast-moving consumer goods (FMCG), and e-commerce.', '8':'Includes aerospace, defense, transportation, construction, and heavy machinery.', '9':'Composed of companies investing in real estate properties and development.', '10':'Covers internet providers, mobile network operators, and satellite communications.', '11':'Includes companies involved in crypto exchanges, blockchain technology, and DeFi.'}
        markets_companies =  {'1': 'Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Google (GOOGL), Amazon (AMZN),...', '2': 'Coca-Cola (KO), Johnson & Johnson (JNJ), IBM, McDonald\'s (MCD), Procter & Gamble (PG),...', '3':'Reliance Industries (India), Alibaba (China), Vale (Brazil), Tata Motors (India),...', '4':'ExxonMobil (XOM), Chevron (CVX), BP, Shell, Saudi Aramco,...', '5':'JPMorgan Chase (JPM), Goldman Sachs (GS), Wells Fargo (WFC), Visa (V), PayPal (PYPL),...', '6':'Pfizer (PFE), Moderna (MRNA), Johnson & Johnson (JNJ), Merck (MRK), Roche (ROG),...', '7':'Walmart (WMT), Amazon (AMZN), Nike (NKE), Procter & Gamble (PG), Tesla (TSLA).', '8':'Boeing (BA), Caterpillar (CAT), Lockheed Martin (LMT), General Electric (GE),...', '9':'Simon Property Group (SPG), Prologis (PLD), Public Storage (PSA),...', '10':'AT&T (T), Verizon (VZ), T-Mobile (TMUS), Vodafone (VOD),...', '11':'Coinbase (COIN), MicroStrategy (MSTR), Bitcoin ETFs (BITO, IBIT), Riot Blockchain (RIOT),...'}

        model_file = request.files.get('model')
        agg_index = request.form.get('agg')
        model_index = request.form.get('model_type')
        cat_index = request.form.get('cat')
        market_index = request.form.get('market_index')
        cat = cats[cat_index]
        client_id = request.form.get('id')
        model_filename = request.form.get('filename')
        region_index = request.form.get('region_index')
        accuracy = request.form.get('accuracy')
        error = request.form.get('error')

        # print(source_models,'\n')
        source_models = os.path.join(source_models,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower(),all_models_name[model_index].replace(" ","-").lower(),agg_methods[agg_index].replace(" ", "-").lower())
        # print(source_models,'\n')
        clients_models = os.path.join(clients_models,markets[market_index].replace(" ", "-").lower(),regions[region_index].replace(" ", "-").lower(),cat.replace(" ", "-").lower(),all_models_name[model_index].replace(" ","-").lower(),agg_methods[agg_index].replace(" ", "-").lower())
        # print(clients_models,'\n')

        if not os.path.exists(source_models):
            os.makedirs(source_models)

        if not os.path.exists(clients_models):
            os.makedirs(clients_models)

        all_client_models = []
        all_client_models_name = []
        accuracies = []
                    
        if model_file:

            client_model_path = os.path.join(clients_models, client_id+'-'+model_filename)
            model_file.save(client_model_path)
            print(client_model_path)

            performance_columns = ['client','model','categorie','aggregation','region','market','accuracy','error','filename']
            new_record = {'client': client_id, 'model': model_index, 'categorie': cat_index, 'aggregation': agg_index, 'region': region_index,'market': market_index , 'accuracy': accuracy, 'error': error, 'filename': client_model_path}
            new_row_df = pd.DataFrame([new_record],columns=performance_columns)

            if os.path.exists('models_performances.csv'):
                df = pd.read_csv('models_performances.csv')
                df = pd.concat([df, new_row_df], ignore_index=True)
            else:
                df = new_row_df #pd.DataFrame(columns=performance_columns)


            df.to_csv('models_performances.csv', index=False)

            df = pd.read_csv('models_performances.csv')
            accuracies = df[(df['categorie'] == cat_index) & (df['model'] == model_index) & (df['aggregation'] == agg_index) & (df['region'] == region_index) & (df['market'] == market_index)]['accuracy']
            all_client_models_name = df[(df['categorie'] == cat_index) & (df['model'] == model_index) & (df['aggregation'] == agg_index) & (df['region'] == region_index) & (df['market'] == market_index)]['filename']
            print(all_client_models_name)


            for file in all_client_models_name:
                file_path = file #os.path.join(clients_models, file)
                if os.path.isfile(file_path):
                    if file_path.endswith('.keras'):
                        c_model = load_model(file_path)
                        all_client_models.append(c_model)
                    elif file_path.endswith('.pkl'):
                        with open(file_path, 'rb') as m_file:
                            c_model = pickle.load(m_file)
                            all_client_models.append(c_model)
                    elif file_path.endswith('.joblib'):
                        c_model = joblib.load(file_path)
                        all_client_models.append(c_model)
        

            if client_model_path.endswith('.joblib'):
                client_model = joblib.load(client_model_path)
                all_client_models.append(client_model)
            elif client_model_path.endswith('.pkl'):
                with open(client_model_path, 'rb') as cl_file:
                    client_model = pickle.load(cl_file)
                    all_client_models.append(client_model)
            elif client_model_path.endswith('.keras'):
                client_model = load_model(client_model_path)
                all_client_models.append(client_model)
            else:
                return jsonify({"error": "Unsupported file type"}), 400


            if len(all_client_models) > 1:

                if agg_index in agg_methods.keys() and agg_index == '4':
                    global_model = aggregate_models_bma(all_client_models,all_models,model_index,accuracies)
                elif agg_index in agg_methods.keys() and agg_index == '3':
                    global_model = aggregate_models_oms(all_client_models,all_models,model_index,accuracies)
                elif agg_index in agg_methods.keys() and agg_index == '2':
                    global_model = aggregate_models_fedma(all_client_models,all_models,model_index)
                else:
                    global_model = aggregate_models_fedavg(all_client_models,all_models,model_index)

            else:
                global_model = client_model

            extension = os.path.splitext(client_model_path)[1]
            global_model_path = os.path.join(source_models, 'global_model' +extension)

            print('\n|>> Global model genereated successfully !\n')
            print(global_model_path)

            if client_model_path.endswith('.joblib'):
                joblib.dump(global_model, global_model_path)
            elif client_model_path.endswith('.pkl'):
                with open(global_model_path, 'wb') as nm_file:
                    pickle.dump(global_model, nm_file)
            elif client_model_path.endswith('.keras'):
                global_model.save(global_model_path)
            else:
                return jsonify({"error": "Unable to save the global model file"}), 400


            return send_file(global_model_path,as_attachment=True)
        

            # os.remove(client_model_path)
        
        else:
            return jsonify({"error": "No model file received!"}), 400


        # return jsonify({"message": {'en':"Parameters received successfully!", 'fr':'Paramètres réçus avec succès!'}}), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "error_message": str(e)}), 500


def aggregate_models_fedavg(models,all_models,model_index):

    aggregated_model = all_models[model_index]

    if model_index == '4':
        #Weights
        weights = [model.get_weights() for model in models]
        avg_weights = [np.mean(np.array([weight[i] for weight in weights]), axis=0) 
                       for i in range(len(weights[0]))]
        #global model
        aggregated_model.set_weights = avg_weights
    elif model_index == '3':
        #Weights
        weights = [model.coefs_ for model in models]
        avg_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*weights)]
        #Intercepts
        intercepts = [model.intercepts_ for model in models]
        avg_intercepts = [np.mean(layer_intercepts, axis=0) for layer_intercepts in zip(*intercepts)]
        #global model
        aggregated_model.coefs_ = avg_weights
        aggregated_model.intercepts_ = avg_intercepts
    else:
        #Weights
        weights = [model.coef_ for model in models]
        avg_weights = np.mean(weights, axis=0)
        #Intercepts
        intercepts = [model.intercept_ for model in models]
        avg_intercepts = np.mean(intercepts, axis=0)
        #global model
        aggregated_model.coef_ = avg_weights
        aggregated_model.intercepts_ = avg_intercepts


    return aggregated_model

def aggregate_models_fedma(models,all_models,model_index):

    aggregated_model = all_models[model_index]

    if model_index == '4':
        #Weights
        weights = [model.get_weights() for model in models]
        avg_weights = [np.mean(np.array([weight[i] for weight in weights]), axis=0) 
                       for i in range(len(weights[0]))]
        #global model
        aggregated_model.set_weights = avg_weights
    elif model_index == '3':
        #Weights
        weights = [model.coefs_ for model in models]
        avg_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*weights)]
        #Intercepts
        intercepts = [model.intercepts_ for model in models]
        avg_intercepts = [np.mean(layer_intercepts, axis=0) for layer_intercepts in zip(*intercepts)]
        #global model
        aggregated_model.coefs_ = avg_weights
        aggregated_model.intercepts_ = avg_intercepts
    else:
        #Weights
        weights = [model.coef_ for model in models]
        avg_weights = np.mean(weights, axis=0)
        #Intercepts
        intercepts = [model.intercept_ for model in models]
        avg_intercepts = np.mean(intercepts, axis=0)
        #global model
        aggregated_model.coef_ = avg_weights
        aggregated_model.intercepts_ = avg_intercepts

    return aggregated_model

def aggregate_models_oms(models,all_models,model_index,accuracies):
    max_accuracy_index = np.argmax(accuracies)
    best_model = models[max_accuracy_index]
    return best_model

def aggregate_models_bma(models,all_models,model_index, accuracies, top_n=3):
    top_indices = np.argsort(accuracies)[0:top_n]
    
    # Initialize an aggregated model (assuming the first model is a template)
    aggregated_model = all_models[model_index]
    
    # Initialize weights for averaging
    total_weights = None
    
    for index in top_indices:
        if model_index == '4':
            model_weights = models[index].get_weights()  # Get weights from each model
        elif model_index == '3':
            model_weights = models[index].coefs_
        else:
            model_weights = models[index].coef_

        if total_weights is None:
            total_weights = np.zeros_like(model_weights)
        
        # Accumulate weights
        for i in range(len(model_weights)):
            total_weights[i] += model_weights[i]
    
    # Average the weights
    average_weights = [weights / top_n for weights in total_weights]
    
    # Set the averaged weights to the aggregated model
    aggregated_model.set_weights(average_weights)
    
    return aggregated_model