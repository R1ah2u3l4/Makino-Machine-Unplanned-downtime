#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('conda install -c anaconda flask')


# In[2]:


import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import re
import pickle, joblib


# In[3]:


impute = joblib.load('medianimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')


# In[4]:


xgb = pickle.load(open('xgb.pkl', 'rb'))


# In[6]:


###Connecting to SQL by creating a sqlachemy engine
from sqlalchemy import create_engine


# In[10]:


get_ipython().system('pip install psycopg2')


# In[11]:


engine = create_engine("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
                       .format(user = "postgres",# user
                               pw = "Rahul@1997", # passwrd
                               db = "postgres")) #database


# In[12]:


app = Flask(__name__)


# In[17]:


import warnings
warnings.filterwarnings("ignore")


# In[18]:


# Define flask


###@app.route('/')
def home():
    return render_template('index.html')

##@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        
        data = pd.read_excel(f)
        
        clean = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)
        
        clean1 = pd.DataFrame(winsor.transform(clean),columns=clean.columns)
        
        clean2 = pd.DataFrame(minmax.transform(clean1))
      
                        
        prediction = pd.DataFrame(xgb.predict(clean2), columns = ['Downtime'])
      
        
        
        
        prediction.to_sql('XGB_test', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        
       
        return render_template("new.html", Y = prediction.to_html(justify='center').replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bordercolor="#000000" bgcolor="#FFCC66">'))


# In[7]:


if __name__=='__main__':
    app.run(debug = True)



# In[ ]:




