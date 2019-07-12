from django.shortcuts import render
from django.http import HttpResponse

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import joblib
import itertools
import requests, json
# Create your views here.

ML_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ml_models')

def Chilli_all_other():
  Rh3 = np.arange(25,50,0.1)
  Rh4 = np.arange(70,100,0.1)
  Rh5 = np.arange(77,85,0.1)
  Rh6 = np.arange(85,100,0.1)
  T3 = np.arange(20,30,0.1)
  T4 = np.arange(22,30,0.1)
  T5 = np.arange(22,25,0.1)
  T6 = np.arange(25,32,0.1)

  # dict = { 0:'Damping Off', 1:'Fruit Rot and Die Back', 2:'Powdery Mildew', 3:'Bacterial Leaf Spot', 4:'Cercospora Leaf Spot', 5:'Fusarium Wilt'}
  # Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

  df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
  df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T',  'Disease'])
  df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T', 'Disease'])
  df6 = pd.DataFrame(data=(list(itertools.product(Rh6,T6,[5]))),columns=['Rh', 'T',  'Disease'])

  df = df3.append(df4.append(df5.append(df6, ignore_index=True),ignore_index=True),ignore_index=True)

  features = ['Rh','T']
  df = df.sample(frac=1).reset_index(drop = True)

  x = df.loc[:,features].values
  y = df.loc[:,['Disease']].values

  x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
  x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

  clf1 = SVC(probability=True).fit(x_tr,y_tr)
  joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Chilli_all_other.sav'))
  return clf1


def Chilli_Flowering():
    Rh2 = np.arange(92,100,0.1)
    Rh3 = np.arange(25,50,0.1)
    Rh4 = np.arange(70,100,0.1)
    Rh5 = np.arange(77,85,0.1)
    Rh6 = np.arange(85,100,0.1)
    T2 = np.arange(27.5,28.5,0.1)
    T3 = np.arange(20,30,0.1)
    T4 = np.arange(22,30,0.1)
    T5 = np.arange(22,25,0.1)
    T6 = np.arange(25,32,0.1)

    # dict = { 0:'Damping Off', 1:'Fruit Rot and Die Back', 2:'Powdery Mildew', 3:'Bacterial Leaf Spot', 4:'Cercospora Leaf Spot', 5:'Fusarium Wilt'}
    # Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

    df2 = pd.DataFrame(data=(list(itertools.product(Rh2,T2,[1]))),columns=['Rh', 'T', 'Disease'])
    df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T',  'Disease'])
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T', 'Disease'])
    df6 = pd.DataFrame(data=(list(itertools.product(Rh6,T6,[5]))),columns=['Rh', 'T',  'Disease'])

    df = df2.append(df3.append(df4.append(df5.append(df6, ignore_index=True),ignore_index=True),ignore_index=True),ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(probability=True).fit(x_tr,y_tr)

    joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Chilli_Flowering.sav'))
    return clf1


def Chilli_Seedling():
    Rh1 = np.arange(90,100,0.1)
    Rh3 = np.arange(25,50,0.1)
    Rh4 = np.arange(70,100,0.1)
    Rh5 = np.arange(77,85,0.1)
    Rh6 = np.arange(85,100,0.1)
    T1 = np.arange(25,35,0.1)
    T3 = np.arange(20,30,0.1)
    T4 = np.arange(22,30,0.1)
    T5 = np.arange(22,25,0.1)
    T6 = np.arange(25,32,0.1)

    # dict = { 0:'Damping Off', 1:'Fruit Rot and Die Back', 2:'Powdery Mildew', 3:'Bacterial Leaf Spot', 4:'Cercospora Leaf Spot', 5:'Fusarium Wilt'}
    # Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

    df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,[0]))),columns=['Rh', 'T', 'Disease'])
    df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T',  'Disease'])
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T', 'Disease'])
    df6 = pd.DataFrame(data=(list(itertools.product(Rh6,T6,[5]))),columns=['Rh', 'T',  'Disease'])

    df = df1.append(df3.append(df4.append(df5.append(df6, ignore_index=True),ignore_index=True),ignore_index=True),ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(probability=True).fit(x_tr,y_tr)

    joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Chilli_Seedling.sav'))
    return clf1


def Tomato_Seedling():
    Rh1 = np.arange(85,100,0.1)
    Rh4 = np.arange(70,100,0.1)
    T1 = np.arange(18,25,0.1)
    T4 = np.arange(25,30,0.1)

    # dict = { 0:'Damping Off', 1:'Septorial Leaf Spot', 2:'Bacterial Stem and Fruit Canker', 3:'Early Blight', 4:'Bacterial Leaf Spot'}
    # Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

    df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,[0]))),columns=['Rh', 'T',  'Disease'])
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T', 'Disease'])

    df = df1.append(df4, ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(probability=True).fit(x_tr,y_tr)

    joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Tomato_Seedling.sav'))
    return clf1

def Tomato_all_others():
    Rh2 = np.arange(75,100,0.1)
    Rh3 = np.arange(75,100,0.1)
    Rh4 = np.arange(70,100,0.1)
    Rh5 = np.arange(80,100,0.1)
    T2 = np.arange(20,25,0.1)
    T3 = np.arange(25,30,0.1)
    T4 = np.arange(25,30,0.1)
    T5 = np.arange(15,21,0.1)

    # dict = { 0:'Damping Off', 1:'Septorial Leaf Spot', 2:'Bacterial Stem and Fruit Canker', 3:'Early Blight', 4:'Bacterial Leaf Spot'}
    # Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

    df2 = pd.DataFrame(data=(list(itertools.product(Rh2,T2,[1]))),columns=['Rh', 'T',  'Disease'])
    df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T', 'Disease'])
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T',  'Disease'])

    df = df2.append(df3.append(df4.append(df5, ignore_index=True),ignore_index=True),ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(kernel='poly', probability=True).fit(x_tr,y_tr)

    joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Tomato_all_others.sav'))
    return clf1

def Cotton_Flowering():

      Rh1 = np.arange(50,80,0.1)
      Rh3 = np.arange(80,100,0.1)
      Rh4 = np.arange(80,100,0.1)
      Rh5 = np.arange(85,100,0.1)

      T1 = np.arange(25,32,0.1)
      T3 = np.arange(29,33,0.1)
      T4 = np.arange(20,30,0.1)
      T5 = np.arange(25,35,0.1)

      df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,[0]))),columns=['Rh', 'T', 'Disease'])
      df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
      df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T',  'Disease'])
      df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T', 'Disease'])

      df = df1.append(df3.append(df4.append(df5, ignore_index=True),ignore_index=True),ignore_index=True)


      features = ['Rh','T']
      df = df.sample(frac=1).reset_index(drop = True)

      x = df.loc[:,features].values
      y = df.loc[:,['Disease']].values
      x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
      x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

      clf1 = SVC(probability=True).fit(x_tr,y_tr)
      joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Cotton_Flowering.sav'))
      return clf1

def Cotton_Seedling():
    Rh1 = np.arange(50,80,0.1)
    Rh2 = np.arange(80,100,0.1)
    Rh4 = np.arange(80,100,0.1)
    Rh5 = np.arange(85,100,0.1)

    T1 = np.arange(25,32,0.1)
    T2 = np.arange(30,42,0.1)
    T4 = np.arange(20,30,0.1)
    T5 = np.arange(25,35,0.1)



    df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,[0]))),columns=['Rh', 'T', 'Disease'])
    df2 = pd.DataFrame(data=(list(itertools.product(Rh2,T2,[1]))),columns=['Rh', 'T', 'Disease'])
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T',  'Disease'])
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T', 'Disease'])


    df = df1.append(df2.append(df4.append(df5, ignore_index=True),ignore_index=True),ignore_index=True)


    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(probability=True).fit(x_tr,y_tr)
    joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Cotton_Seedling.sav'))
    return clf1


def Cotton_all_others():
    Rh1 = np.arange(80,100,0.1)
    Rh3 = np.arange(80,100,0.1)
    Rh5 = np.arange(85,100,0.1)
    T1 = np.arange(28,32,0.1)
    T3 = np.arange(20,30,0.1)
    T5 = np.arange(30,40,0.1)

    df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,[0]))),columns=['Rh', 'T', 'Disease'])
    df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T',  'Disease'])
    df = df1.append(df3.append(df5, ignore_index=True),ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(probability=True).fit(x_tr,y_tr)
    joblib.dump(clf1, os.path.join(ML_MODEL_DIR,'Cotton_all_others.sav'))
    return clf1


try:
    chilli_all_other = joblib.load(os.path.join(ML_MODEL_DIR,'Chilli_all_other.sav'))
except:
    chilli_all_other = Chilli_all_other()
try:
    chilli_flowering = joblib.load(os.path.join(ML_MODEL_DIR,'Chilli_Flowering.sav'))
except:
    chilli_flowering = Chilli_Flowering()
try:
    chilli_seedling = joblib.load(os.path.join(ML_MODEL_DIR, 'Chilli_Seedling.sav'))
except:
    chilli_seedling = Chilli_Seedling()
try:
    tomato_seedling = joblib.load(os.path.join(ML_MODEL_DIR, 'Tomato_Seedling.sav'))
except:
    tomato_seedling = Tomato_Seedling()
try:
    tomato_all_other = joblib.load(os.path.join(ML_MODEL_DIR, 'Tomato_all_others.sav'))
except:
    tomato_all_other = Tomato_all_others()
try:
    cotton_all_other = joblib.load(os.path.join(ML_MODEL_DIR, 'Cotton_all_others.sav'))
except:
    cotton_all_other = Cotton_all_others()
try:
    cotton_flowering = joblib.load(os.path.join(ML_MODEL_DIR, 'Cotton_Flowering.sav'))
except:
    cotton_flowering = Cotton_Flowering()
try:
    cotton_seedling = joblib.load(os.path.join(ML_MODEL_DIR, 'Cotton_Seedling.sav'))
except:
    cotton_seedling = Cotton_Seedling()


def predict(Crop, Stage, l):
    temp = []
    temp1=[]
    i=0
    if(Crop=='Hot Pepper (Chilli)'):
        map = ['Damping Off', 'Fruit Rot and Die Back', 'Powdery Mildew', 'Bacterial Leaf Spot', 'Cercospora Leaf Spot', 'Fusarium Wilt']
        if(Stage=='Seedling'):
            for var in l:
                temp.append(chilli_seedling.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
              var=var[0].tolist()
              var=[var[0]]+[0]+var[1:]
              temp1.insert(i,var)
              i=i+1

        elif (Stage=='Flowering'):
            for var in l:
                temp.append(chilli_flowering.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
              var=var[0].tolist()
              var=[0]+var
              temp1.insert(i,var)
              i=i+1
        else:
            for var in l:
                temp.append(chilli_all_other.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
              var=var[0].tolist()
              var=[0,0]+var
              temp1.insert(i,var)
              i=i+1
    elif(Crop=='Tomato'):
        map = [ 'Damping Off', 'Septorial Leaf Spot', 'Bacterial Stem and Fruit Canker', 'Early Blight', 'Bacterial Leaf Spot']
        if(Stage=='Seedling'):
            for var in l:
                temp.append(tomato_seedling.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
              var=var[0].tolist()
              var=[var[0]]+[0,0]+[var[1]]+[0]
              temp1.insert(i,var)
              i=i+1
        else:
            for var in l:
                temp.append(tomato_all_other.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
              var=var[0].tolist()
              var=[0]+var
              temp1.insert(i,var)
              i=i+1
    elif(Crop=='Cotton'):
        map = ['Fusarium Wilt', 'Root Rot', 'Anthracnose', 'Alternia Leaf Spot', 'Bacterial Blight']
        if(Stage=='Seedling'):
            for var in l:
                temp.insert(i,cotton_seedling.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
                var=var[0].tolist()
                var=[var[0],var[1]]+[0]+var[2:]
                temp1.insert(i,var)
                i=i+1
        elif (Stage=='Flowering'):
            for var in l:
                temp.insert(i,cotton_flowering.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
                var=var[0].tolist()
                var=[var[0]]+[0]+var[1:]
                temp1.insert(i,var)
                i=i+1
        else:
            for var in l:
                temp.insert(i,cotton_all_other.predict_proba(np.asarray(var).reshape(1,-1)))
            for var in temp:
                var=var[0].tolist()
                var=[var[0]]+[0,0]+[var[1]]+[0]+[var[2]]
                temp1.insert(i,var)
                i=i+1
    return temp1, map

def index(request):
    js = {}
    try:
        crop = js['crop'] =  request.GET.get('crop')
        stage = js['stage'] = request.GET.get('stage')
        lat = js['lat']  = request.GET.get('lat')
        lng = js['lng'] = request.GET.get('lng')
        url = 'https://weather.api.here.com/weather/1.0/report.json?app_id=myawyUf5nPfe914BSGfp&app_code=F2A2ZPi1ETd6lGpeBLmJqA&product=forecast_hourly&oneobservation=true' + '&latitude=' + str(lat) + '&' + 'longitude=' + str(lng)
        response = requests.get(url)
        js_data = response.json()
        js_data = js_data.get('hourlyForecasts').get('forecastLocation').get('forecast')
        i=0
        Tavg=[]
        Uavg=[]
        RHavg=[]
        for j in range(7):
            t=[]
            rh=[]
            u=[]
            for dict in js_data[i:i+24]:
                t = t + [float(dict['temperature'])]
                rh = rh + [float(dict['humidity'])]
                u = u + [float(dict['windSpeed'])]
            Tavg=Tavg+[round(sum(t)/24, 1)]
            RHavg=RHavg+[round(sum(rh)/24, 2)]
            Uavg=Uavg+[round(sum(u)/24,1)]
            i=i+24
        z = list(zip(RHavg, Tavg))
        js['prob'], js['map'] = predict(crop, stage, z)
        js['status'] = True
    except Exception as e:
        js['status'] = False
        js['error'] = str(e)
    return HttpResponse(json.dumps(js),content_type="application/json")


def esp(request):
    response = {}
    data = request.GET.get("data")
    if data:
        f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'soil.txt'), 'a+')
        f.write(str(datetime.datetime.now().strftime('%H:%M%S, %d-%m-%Y')) + ": " + data + '\n')
        f.close()
        response['status'] = True
        response['message'] = "Data submitted successfully."
    else:
        response['status'] = False
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'soil.txt'), 'r')
    data = f.read().split('\n')
    data.reverse()
    if len(data)>=5:
        context['data'] = data[:5]
    else:
        context['data'] = data
    return HttpResponse(json.dumps(response), content_type="application/json")
