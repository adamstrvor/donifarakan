import kagglehub
import os
import sys
import shutil
from  termcolor import colored

source_dir = os.path.dirname(__file__)

sites = {'1':'kaggle'}
cats = {'1':'price','2':'news','3':'risk'}

print("--------------------------------------")
print(" GET YOUR DATASET")
print("--------------------------------------")

print('|> Please select the repository:')
for index,name in sites.items():
	print(index,". "+name)

site_index = input("").strip()
site = sites[site_index] if site_index in sites.keys() else "kaggle"
print(colored(f"\t|>> {site} \n",'blue'))

print('|> Provide the dataset link:')
data_link = input("").strip()
data_link = data_link if data_link != "" else "mayankanand2701/tesla-stock-price-dataset"
print(colored(f"\t|>>  {data_link}\n",'blue'))

print('|> Specify the categorie of the dataset [price,news,risk,...]:')
for index,name in cats.items():
	print(index,". "+name)
cat_index = input("").strip()
cat = cats[cat_index] if cat_index in cats.keys() else "general"
print(colored(f"\t|>> {cat}\n",'blue'))

source_dataset = os.path.join(source_dir,cat)

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

