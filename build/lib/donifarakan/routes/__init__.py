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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import joblib
import pickle
import requests


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

        cat = request.form.get('cat')

        source_models = os.path.join(source_models,cat)
        # print(source_models,'\n')

        if not os.path.exists(source_models):
            os.makedirs(source_models)

        if os.path.exists(os.path.join(source_models, 'global_model.joblib')):
            return send_file(os.path.join(source_models, 'global_model.joblib'),as_attachment=True)
        elif os.path.exists(os.path.join(source_models, 'global_model.pkl')):
            return send_file(os.path.join(source_models, 'global_model.pkl'),as_attachment=True)
        elif os.path.exists(os.path.join(source_models, 'global_model.keras')) :
            return send_file(os.path.join(source_models, 'global_model.keras'),as_attachment=True)
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
        clients_models = os.path.join(source_models,'clients')

        cats = {'1':'price','2':'news','3':'risk'}
        agg_methods = {'1':"Federated Averaging (FedAvg)", '2':"Federated Matched Averaging (FedMA)", '3':"All Model Averaging (AMA)", '4': "One Model Selection (OMS)", '5':"Best Models Averaging (BMA)", '6': "FedProx", '7': "Hybrid Approaches"}
        all_models = { '1': LinearRegression(), '2': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500), '3': Sequential() }
        all_models_name = {'1': 'Linear Regression Model', '2': 'Mutli-Layer Perceptron (MLP)', '3': 'Long-Short Term Memory (LSTM)'}
        regions = {'1': 'Africa', '2': 'America', '3': 'Middle east', '4': 'Europe', '5': 'Asia'}

        model_file = request.files.get('model')
        agg_index = request.form.get('agg')
        model_index = request.form.get('model_type')
        cat = request.form.get('cat')
        client_id = request.form.get('id')
        model_filename = request.form.get('filename')
        region_index = request.form.get('region_index')
        accuracy = request.form.get('accuracy')
        error = request.form.get('error')

        # print(source_models,'\n')
        source_models = os.path.join(source_models,cat)
        # print(source_models,'\n')
        clients_models = os.path.join(clients_models,cat)
        # print(clients_models,'\n')

        if not os.path.exists(source_models):
            os.makedirs(source_models)

        if not os.path.exists(clients_models):
            os.makedirs(clients_models)

        all_client_models = []
        accuracies = []
        for file in os.listdir(clients_models):
                file_path = os.path.join(clients_models, file)
                if file.startswith('.') == False and os.path.isfile(file_path):
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
                    
        if model_file:
            client_model_path = os.path.join(clients_models, client_id+'-'+model_filename)

            model_file.save(client_model_path)

            if os.path.exists('models_performances.csv'):
                df = pd.read_csv('models_performances.csv')
                accuracies = df['accuracy']
            else:
                df = pd.DataFrame(columns=['client','model','categorie','region','accuracy','error'])
                accuracies.append(accuracy)

            print(client_model_path)

            new_record = {'client': client_id, 'categorie': cat, 'region': regions[region_index], 'model': all_models_name[model_index], 'accuracy': accuracy, 'error': error}
            new_row_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_row_df], ignore_index=False)
            df.to_csv('models_performances.csv', index=True)

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
            global_model_path = os.path.join(source_models, 'global_model'+extension)

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
    # Initialize an empty MLPRegressor for aggregation
    aggregated_model = all_models[model_index] #MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500) 

    # Get the weights of all models and average them
    weights = [model.coefs_ for model in models]
    avg_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*weights)]
    aggregated_model.coefs_ = avg_weights

    if hasattr(models[0], 'intercepts_'):
        intercepts = [model.intercepts_ for model in models]
        avg_intercepts = [np.mean(layer_intercepts, axis=0) for layer_intercepts in zip(*intercepts)]
        aggregated_model.intercepts_ = avg_intercepts

    return aggregated_model

def aggregate_models_fedma(models,all_models,model_index):
    # Initialize an empty MLPRegressor for aggregation
    aggregated_model = all_models[model_index] #MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)

    # Get the weights of all models
    weights = [model.coefs_ for model in models]
    avg_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*weights)]
    aggregated_model.coefs_ = avg_weights

    if hasattr(models[0], 'intercepts_'):
        intercepts = [model.intercepts_ for model in models]
        avg_intercepts = [np.mean(layer_intercepts, axis=0) for layer_intercepts in zip(*intercepts)]
        aggregated_model.intercepts_ = avg_intercepts


    return aggregated_model

def aggregate_models_oms(models,all_models,model_index,accuracies):
    max_accuracy_index = np.argmax(accuracies)
    best_model = models[max_accuracy_index]
    return best_model

def aggregate_models_bma(models,all_models,model_index, accuracies, top_n=3):
    top_indices = np.argsort(accuracies)[0:top_n]
    
    # Initialize an aggregated model (assuming the first model is a template)
    aggregated_model = models[top_indices[0]].__class__()
    
    # Initialize weights for averaging
    total_weights = None
    
    for index in top_indices:
        model_weights = models[index].get_weights()  # Get weights from each model
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