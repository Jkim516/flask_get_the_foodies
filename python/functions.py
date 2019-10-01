import pandas as pd
import numpy as np
import random

from selenium.webdriver import Chrome
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from lxml import html 
import requests

import time
from time import sleep

import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import shap

shap.initjs()

def my_whole_project(url):
    url_list = get_urls(url)
    df = get_df(url_list)
    shap_graph = get_shap(df)
    f1_score = get_scr(df)
    return shap_graph, f1_score

def df_indi(url):
    url_list = get_urls(url)
    df = get_df(url_list)
    return df

def f1_cm(url):
    url_list = get_urls(url)
    df = get_df(url_list)
    f1_score = get_scr(df)
    return f1_score

def get_urls(url):
    
    """The function generates the most recent 10 page of reviews on Yelp for the individual restaurant."""
    browser = Chrome()
    url_list = []
    
    link = url+'?&sort_by=date_desc' #for the first page of the latest 20 reviews/ratings
    url_list.append(link)
    
    for num in range(20, 200, 20): #for the 2nd to 10th page of the latest reviews/ratings
        links = url+'?'+'&start='+str(num)+'&sort_by=date_desc'
        url_list.append(links)
        
    return url_list

def get_df(url_list):
    
    """The output of the function is a dataframe that includes the full texts and ratings for the
       latest 200 customer reviews on Yelp for the particular restaurant."""
    
    ratings = []
    reviews = []
    for url in url_list:
        browser.get(url)
        time.sleep(5 + random.random()*5)
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        #get ratings
        xpaths = []
        ratings_xp = range(1, 21)
        try: 
            for num in ratings_xp:
                xpath = '//*[@id="wrap"]/div[3]/div/div[1]/div[3]/div/div/div[2]/div[1]/section[5]/div/section[2]/div[2]/div/ul/li['+str(num)+']/div/div[2]/div[1]/div/div[1]/span/div'
                xpaths.append(xpath)
        except: 
            for num in ratings_xp:
                xpath = '//*[@id="wrap"]/div[3]/div/div[1]/div[3]/div/div/div[2]/div[1]/section[6]/div/section[2]/div[2]/div/ul/li['+str(num)+']/div/div[2]/div[1]/div/div[1]/span/div'
                xpaths.append(xpath)
        for ans in xpaths:
            rating_ind = WebDriverWait(browser, 20).until(EC.visibility_of_element_located((By.XPATH, ans))).get_attribute("aria-label")
            ratings.append(rating_ind)

        #get reviews    
        lemons = soup.find_all('p', class_="lemon--p__373c0__3Qnnj text__373c0__2pB8f comment__373c0__3EKjH text-color--normal__373c0__K_MKN text-align--left__373c0__2pnx_")
        for lemon in lemons:
            reviews.append(lemon.text)

    #remove 'star ratings'
    ratings = [val.replace(' star rating','') for val in ratings]
    ratings = [int(val) for val in ratings]
    
    #combine as df        
    merged_list = [(ratings[i], reviews[i]) for i in range(0, len(ratings))]
    df = [{'rating': rating, 'text': review} for rating, review in merged_list]
    res_df = pd.DataFrame(df)
    
    #group ratings >= 4 as positive ratings and others as negative ratings 
    def ratings(rf):
        if rf['rating'] >= 4:
            return 1
        else:
            return 0
    res_df['pos_neg'] = res_df.apply(ratings, axis=1)
    
    return res_df 

def get_shap(res_df):     
    
    """The model is trained with the latest 10 page of reviews on Yelp."""
    
    #train test split
    features = res_df['text']
    target = res_df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    #vec
    vectorizer = TfidfVectorizer(stop_words='english')
    
    #vec for SHAP
    features_train_transformed = vectorizer.fit_transform(X_train)
    features_test_transformed = vectorizer.transform(X_test)    
    
    # Fit a linear logistic regression model
    model = LogisticRegression(solver='lbfgs')
    model.fit(features_train_transformed, y_train)
    
    # Explain the linear model
    explainer = shap.LinearExplainer(model, features_train_transformed, feature_dependence="independent")
    shap_values = explainer.shap_values(features_test_transformed)
    X_test_array = features_test_transformed.toarray() 
    
    fig, ax = plt.subplots(figsize=((14, 7)))
    shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names(), show=False, auto_size_plot=False)
#     fig.savefig("/Users/Erica/flatiron/flask_app/capstone-flask-app-template-seattle-ds-062419/static/images/shap_plot.png", pad_inches=.1)
    
    # Summarize the effect of all the features                         
    return fig.savefig("/static/images/shap_plot.png", pad_inches=.1)

def get_scr(res_df):

    """The output shows the F1 score and confusion matrix of the model."""
    
    #train test split
    features = res_df['text']
    target = res_df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    #vec
    vectorizer = TfidfVectorizer(stop_words='english')
    
    #get confusion matrix and f1 score
    classifier = LogisticRegression(solver='lbfgs')
    pipe = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    predicted_labels = pipe.predict(X_test)
        
    f1 = f1_score(y_test, predicted_labels)
    cm = confusion_matrix(y_test, predicted_labels)
    chart = pd.DataFrame(cm, 
                         index = ['Actual Negative', 'Actual Positive'], 
                         columns=['Predicted Negative', 'Predicted Positive'])

    return ('F1', f1), chart

