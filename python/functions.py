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
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import shap

shap.initjs()

def full_shap_eval(url, ind=0):
    
    """The function returns the summary plot of the model along with f1 score and confusion matrix."""
    
    url_list = get_urls(url)
    df = get_df(url_list)
    f1_score = get_f1scr(df)
    #draw = force_plot(df, ind)
    txt = force_plot_text(df, ind)

    return f1_score, txt

def get_pos_neg_words_df(url):
    
    """The function returns the dataframe includes both 
    'Words with Positive Impact on Ratings' and 'Words with Negative Impact on Ratings.'"""
    
    url_list = get_urls(url)
    df = get_df(url_list)
    words = show_aud(df)
    
    return words

def f1_ind(url):
    
    """The function returns f1 Score of the model"""
    
    url_list = get_urls(url)
    df = get_df(url_list)
    f1_score = get_f1scr(df)
    return f1_score


def force_plot_ind(url, ind=0):
    
    """graph the force plot for the review"""
    
    url_list = get_urls(url)
    df = get_df(url_list)
    draw = force_plot(df, ind)
    return draw

def force_text_ind(url, ind=0):
    
    """graph the force plot for the review"""
    
    url_list = get_urls(url)
    df = get_df(url_list)
    txt = force_plot_text(df, ind)
    return txt

def get_urls(url):
    
    """The function generates the most recent 10 page of reviews on Yelp for the individual restaurant."""
    
    url_list = []
    
    link = url+'?&sort_by=date_desc' #for the first page of the latest 20 reviews/ratings
    url_list.append(link)
    
    for num in range(20, 60, 20): #for the 2nd to 10th page of the latest reviews/ratings
        links = url+'?'+'&start='+str(num)+'&sort_by=date_desc'
        url_list.append(links)
        
    return url_list

def get_df(url_list):
    
    """The output of the function is a dataframe that includes the full texts and ratings for the
       latest 200 customer reviews on Yelp for the particular restaurant."""
    
    browser = Chrome()

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

def show_aud(df):
    
    """the function returns two list of words that are discussed on Yelp about the particular restaurant."""
    
    #train test split
    features = df['text']
    target = df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    #vec
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['highly','amazing','great','did','make','wa','don', 'didn','oh','ve','definitely','absolutely','cool'])
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    
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
    
    #build dataframe for illustration 
    res_sv_df = pd.DataFrame(shap_values, columns=vectorizer.get_feature_names())
    
    #present max/min SHAP values for 
    neg_df = res_sv_df.min(axis=0)
    neg_sort = neg_df.sort_values().head(15)
    neg_list = list(dict(neg_sort).keys())
    neg_df_final = pd.DataFrame(neg_list, index=range(1,16), columns=['Words with Negative Impact on Ratings'])
    
    pos_df = res_sv_df.max(axis=0)
    pos_sort = pos_df.sort_values(ascending=False).head(15)
    pos_list = list(dict(pos_sort).keys())
    pos_df_final = pd.DataFrame(pos_list, index=range(1,16), columns=['Words with Positive Impact on Ratings'])
    
    combo_df = pd.concat([pos_df_final, neg_df_final], axis=1)
    
    return combo_df

def get_f1scr(res_df):

    """The output shows the rounded F1 score."""
    
    #train test split
    features = res_df['text']
    target = res_df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    #vec
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['highly','amazing','great','did','make','wa','don', 'didn','oh','ve','definitely','absolutely','cool'])
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    
    #get confusion matrix and f1 score
    classifier = LogisticRegression(solver='lbfgs')
    pipe = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    predicted_labels = pipe.predict(X_test)
        
    f1 = f1_score(y_test, predicted_labels)
    f1_scr = 'F1 Score: {}'.format(round(f1,3))
    
    return f1_scr

def get_cm(res_df):

    """The output shows the dataframe of the confusion matrix."""
    
    #train test split
    features = res_df['text']
    target = res_df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    #vec
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['highly','amazing','great','did','make','wa','don', 'didn','oh','ve','definitely','absolutely','cool'])
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    
    #get confusion matrix and f1 score
    classifier = LogisticRegression(solver='lbfgs')
    pipe = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    predicted_labels = pipe.predict(X_test)
        
    cm = confusion_matrix(y_test, predicted_labels)
    chart = pd.DataFrame(cm, 
                         index = ['Actual Negative', 'Actual Positive'], 
                         columns=['Predicted Negative', 'Predicted Positive'])

    return chart

def force_plot(res_df, ind=0):
    
    #train test splits
    features = res_df['text']
    target = res_df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    #vec
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['highly','amazing','great','did','make','wa','don', 'didn','oh','ve','definitely','absolutely','cool'])
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    
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
    
    #draw force plot
    #fig = plt.figure()
    force = shap.force_plot(explainer.expected_value, shap_values[ind,:], X_test_array[ind,:], feature_names=vectorizer.get_feature_names(), show=False, matplotlib=True,figsize=(100, 50))
    #fig.savefig('/static/images/shap_plot.png', pad_inches=.1)
    #print("force is " + str(type(force)))

    return force

def force_plot_text(res_df, ind=0):
    
    """a graph to present the force plot for a review."""
    
    #train test split
    features = res_df['text']
    target = res_df['pos_neg']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
       
    #show text
    for ind in y_test.values: 
        if y_test.values[ind] == 1:
            return('Positive Review:', X_test.values[ind])
        else:
            return('Negative Review:', X_test.values[ind])