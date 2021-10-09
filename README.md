# Starbucks Capstone Project


## Project Overview:
The Udacity Data Scientist Nanodegree Capstone project dataset is a simulation of customer behavior on the Starbucks rewards mobile application. The dataset contains details about three offers made by starbucks namely informational offer (which is an advertisement about a product), discount offer, and BOGO offer (Buy 1 Get 1 Free). An important point to take note is that that not all users receive the same offer. In addition to that the simulated dataset is based on only one product.

The simulated dataset provided by Starbucks is having 3 files. The first file is "portfolio.json" which contains the details about the various offers. The second file is "profile.json" and it contains the demographic data of the Starbucks customers. The third file is "transcript.json" which contains details about the transactions performed by the customers.

An offer is considered completed only when a customer views the offers and performs transactions of the required amount within the stipulated duration.

## Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


## Problem Statement

The main motive of this project is to build a model to see whether the people who viewed the offer will complete the offer.

The following steps are performed to achieve the goal.

1. Exploration Data Analysis
2. Data Cleaning and Data Processing
3. Performing Data Analysis on the Final Dataset
4. Data Modeling
5. Trying various Supervised Learning Models
6. Evalutaion of the Models
7. Model Refinement (Implementing gridsearchcv to find the best parameters of the chosen model if the results are required to be improved)


## Installations:

The below libraries are used in this project:

1.) Data Science Libraries
import pandas as pd
import numpy as np
import math
import json
import datetime

2.) For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

3.) To Serialize and de-serialize a Python object structure
import pickle

4.) To render pandas dataframe in Jupyter Notebook
import qgrid

5.) Interactive HTML widgets for Jupyter Notebooks
import ipywidgets as widgets

6.) Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

## File Descriptions:

The project contains the following files and directories:

	1. Starbucks_Capstone_notebook.ipynb: Jupyter notebook that contains the code related to this project

	2. data: directory containing the json datasets

	3. pic1.png: image file related to the jupyter notebook
		
	4. pic2.png: image file related to the jupyter notebook


## Instructions to run the project:
To execute the code, please follow the below steps in your local system
1. Upload the files and folder available in the GitHub repository to the Jupyter Notebook Workspace
2. Unzip the file transcript.json in the data directory
2. Open the Jupyter Notebook - Starbucks_Capstone_notebook.ipynb and execute all the cells

## Results Summary:
1. Starbucks used the email platform to send the maximum number of offer notifications to the customers.
2. The second most popular channel for sending the offer notifications is mobile. It is followed by web and social platforms. This is the pattern followed by the Starbucks irrespective of the year.
3. BOGO and the discount offer types are the most common offer types. Informational offer is the least common offer type.
4. The most common offer type is BOGO for both new and regular memberships. However, the most common offer type is discount for the loyal membership.
5. Among all the age-categories, the older-adult (60-105) group of customers are having the maximum number of people in the high (90000-130000) income range.
6. The adult (35-60) group of cusomers are having the maximum number of people in both above-average (60000-90000) and average (25000-60000) income range.
7. The average duration it takes for a customer to perform 2 transactions is around 15.48 days.
8. The total number of offers completed by both male and female customers are 31943 out of the 49087 offers they viewed.
9. Nearly 65.07 % of the customers that viewed the offers completed it. Females completed nearly 74.46 % of the offers they viewed. However, males completed only 58.18 % of the offers they viewed.

## Acknowledgements:
Thanks to Starbucks for providing the dataset used in this project and, Udacity for giving the oppurtunity to work on this project.