from transformers import AutoModel, AutoTokenizer
from ctypes.wintypes import tagSIZE
import numpy as np
import pandas as pd
from scipy.fft import idct
import torch
import pymongo
from newscatcherapi import NewsCatcherApiClient
from bson import json_util
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request, jsonify, json, Response
import requests
from bs4 import BeautifulSoup
import pandas as pd
import bson.json_util as json_util
from bson.json_util import dumps,RELAXED_JSON_OPTIONS
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
import flask
import regex as re
from datetime import datetime


def bs4articleextractor(url):
    r = requests.get(url)
    html_content=r.content
    soup=BeautifulSoup(html_content, "html.parser")
    try:
      article = soup.find("div", {"class": "content_wrapper arti-flow"}).get_text()
    except:
      article = " "
    return article

def bs4tagextractor(url):
    r = requests.get(url)
    html_content=r.content
    soup=BeautifulSoup(html_content, "html.parser")
    try:
      taggings = soup.find("div", {"class":"tags_first_line"}).get_text()
      tags = re.findall(r'(?<=#)\w+', taggings)
    except:
      tags = " "
    return tags

tokenizer = AutoTokenizer.from_pretrained("rajeeva703/news_sentiment_rajeev")
model = AutoModel.from_pretrained("rajeeva703/news_sentiment_rajeev")


available_calls_per_key = 1000
available_calls_per_key_per_month = available_calls_per_key // 30

def return_catcherAPI():
    with open("for_app.json", 'r+') as file:
        left_calls = json.load(file)
       
        for key in left_calls["keys"]:
            if left_calls["keys"][key] >= available_calls_per_key_per_month:
                left_calls["keys"][key] -= 1
                print(key)
                break
        left_calls["updated"] = datetime.now().strftime("%d-%m-%Y")
        if datetime.now().date().day != 1:
            left_calls["updated_on_1st"] = False
        if datetime.now().date().day == 1 and left_calls["updated_on_1st"] == False:
            for key in left_calls["keys"]:
                left_calls["keys"][key] = available_calls_per_key_per_month
               
            left_calls["updated_on_1st"] = True
       
        file.seek(0)
        file.truncate()
        json.dump(left_calls, file)
    return NewsCatcherApiClient(x_api_key=key)

mongo_ip_local = "mongodb://localhost:27017/"
# mongo_ip_local = "mongodb://host.docker.internal:27017/"
# mongo_ip_server = "mongodb://10.0.5.193:27017/"

myclient = pymongo.MongoClient(mongo_ip_local)
mydb = myclient["pulsex_db"]
mycol = mydb["Newscol"]

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/update', methods=['POST'])
def final():

  newscatcherapi = return_catcherAPI()

  a1 = newscatcherapi.get_search_all_pages(q= 'Stocks' ,
                                         sources='moneycontrol.com',
                                         lang='en',
                                         countries='IN',
                                         page_size=10,
                                         max_page=1,
                                         seconds_pause=1.0
                                         )

  # a1['articles']
  headline = []
  for i in range(len(a1['articles'])):
    headline.append(a1['articles'][i]['title'])
  headline

  link = []
  for i in range(len(a1['articles'])):
    link.append(a1['articles'][i]['link'])
  link

  id = []
  for i in range(len(a1['articles'])):
    id.append(a1['articles'][i]['_id'])
  id

  score = []
  for i in range(len(a1['articles'])):
    score.append(a1['articles'][i]['_score'])
  score

  author = []
  for i in range(len(a1['articles'])):
    author.append(a1['articles'][i]['authors'])
  author

  website = []
  for i in range(len(a1['articles'])):
    website.append(a1['articles'][i]['clean_url'])
  website

  country = []
  for i in range(len(a1['articles'])):
    country.append(a1['articles'][i]['country'])
  country

  excerpt = []
  for i in range(len(a1['articles'])):
    excerpt.append(a1['articles'][i]['excerpt'])
  excerpt

  opinion = []
  for i in range(len(a1['articles'])):
    opinion.append(a1['articles'][i]['is_opinion'])
  opinion

  language = []
  for i in range(len(a1['articles'])):
    language.append(a1['articles'][i]['language'])
  language

  media = []
  for i in range(len(a1['articles'])):
    media.append(a1['articles'][i]['media'])
  media

  date = []
  for i in range(len(a1['articles'])):
    date.append(a1['articles'][i]['published_date'])
  date

  rights = []
  for i in range(len(a1['articles'])):
    rights.append(a1['articles'][i]['rights'])
  rights

  summary = []
  for i in range(len(a1['articles'])):
    summary.append(a1['articles'][i]['summary'])
  summary

  topic = []
  for i in range(len(a1['articles'])):
    topic.append(a1['articles'][i]['topic'])
  topic

  twitter = []
  for i in range(len(a1['articles'])):
    twitter.append(a1['articles'][i]['twitter_account'])
  twitter



  
  labels = {0:'Neutral', 1:'Negative',2:'Positive'}

  result = []
  count=0
  for x in headline:
      inputs = tokenizer(x, return_tensors="pt", padding=True)
      outputs = model(**inputs)
      # print(outputs)
      val = labels[np.argmax(outputs[0].detach().numpy())]
      probs = torch.nn.functional.softmax(outputs.logits, dim = -1)
      array = probs.detach().numpy()
      new_arr = np.round_(array[0], decimals = 3)
      # Dict = {'Sentence': x, 'Label': val,'Score':str(np.max(new_arr)), 'Other_scores': {'Neutral':str(new_arr[0]),'Negative':str(new_arr[1]),'Positive':str(new_arr[2])}, 'News_details':a1['articles'][count]}
      Dict = {'Headline': x, 'Label': val,'Score':str(np.max(new_arr)), 'Other_scores': {'Neutral':str(new_arr[0]),'Negative':str(new_arr[1]),'Positive':str(new_arr[2])}, 'Link':link[count], "ID": id[count],"relevancy_score": score[count], "Author": author[count], "Website":website[count], "Country": country[count], "Excerpt": excerpt[count], "Is_Opinion":opinion[count], "Language":language[count],"Media":media[count], "Published Date": date[count], "Rights": rights[count], "First_250_words":summary[count],"Topic": topic[count], "Twitter Account": twitter[count], "Tags": bs4tagextractor(link[count]), "Article": bs4articleextractor(link[count])}
      # print(a1['articles'][count])
      result.append(Dict)
      print(result)
      count = count + 1
  x = mycol.insert_many(result)
  return jsonify (json_util.dumps(result))

scheduler = BackgroundScheduler()
scheduler.add_job(func=final, trigger="interval", minutes=60)
scheduler.start()



@app.route("/mgresult")
def home():
    todos = mycol.find()
    json_docs = [json.dumps(doc, default=json_util.default) for doc in todos]
    return flask.jsonify(json_docs)


# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())
if __name__ == "__main__":
  app.run(debug=True)