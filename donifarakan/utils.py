import pandas as pd
import kagglehub
import os
import sys
import shutil
from  termcolor import colored

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

    print('|> Please select the repository:')
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

    print('|> Specify the type or categorie of the dataset (stock price data, feedback data,...):')
    print('\n\t|>> ',end="")
    cat = input("").strip()
    cat = cat if cat != "" else "default"
    cat = cat.replace(" ", "-")
    print(colored(f"\t|>> {cat}\n",'blue'))

    source_dataset = os.path.join(dataset_path,cat)

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



