## Introduction

**"Dɔni farakan"** is a robust federated learning framework designed to enable distributed machine/deep learning while safeguarding data privacy. This framework enable multiple clients to collaboratively train sophisticated models without exposing their raw data. Instead, clients share model updates (e.g., weights) with a central server. The server aggregates these updates to create a generalized and high-performing model.

Specifically tailored for the finance sector (banks, fintech companies, etc.), Dɔnifarakan allows stakeholders to train models on their local data without compromising sensitive information. This collaborative approach facilitates a wide range of applications, including:

- Making accurate predictions (on stock trends,...)
- Preventing market risks.
- Assessing the impact of news on stock market.
- ....

> Follow the steps below to understand how does it works and start creating your own plateform where companies might susbcribe as client in order to use it!

## Requirements

Before to start you need to have **python** installed on your laptop. If you don't have python please follow the steps on their [website](https://www.python.org/downloads) in order to get it.

## Getting started

As we are in the context of federated learning, you should have at least two computers. One representing the central server and another one representing a sample client. You can many clients and as many as you can. Make sure that the computers including central and clients are all on the same local network.

**On your central server computer**

1. Create a new folder for your project. You can name it " MyFramework "
2. Go the folder directory in your terminal or cmd. _$cd "MyFramework"_
3. Install the donifarakan framework. `$ pip install donifarakan`
4. Create a python file which will be used to launch your central server. `$ touch server.py`
5. Open the _server.py_ file then copy and paste this code below:

```
# Here we are loading the required methods from donifrakan
from donifarakan.server import start

# Here we launching the central server
start(ip_address="10.12.167.82",port=6590)

```

**On your client device**

1. Create a new folder for your project. You can name it " MyCompany AI "
2. Go the folder directory in your terminal or cmd. _$cd "MyCompany AI"_
3. Install the donifarakan framework. `pip install donifarakan`
4. Create a python file which will be used to train a model on the client local data. `$touch train.py`
5. Open the _train.py_ file then copy and paste this code below:

```
# Here we are loading the required methods from donifrakan
from donifarakan.utils import download_dataset
from donifarakan.client import global_model, train

# If you don't have any dataset yet, execute this method to download a sample dataset, by default it will download a stock price dataset if you don't specify any!
download_dataset()

# Here you will be requested to provide your central server details in order to get the global model
get_global_model()

# Here you will be requested to provide all the details about your dataset before to start the training
train()
```


## Contribution

> This project is an open source, to contribute on it fell free to explore the architecture and leave your comments and suggestions!

## FAQ

For any query you can reach me out at ___adamaseydoutraore86@gmail.com___

## LICENCE

MIT License

**Copyright (c) 2025 TRAORE ADAMA SEYDOU**
