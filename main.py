#!/usr/bin/env python
# coding: utf-8

# In[303]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
data_test = pd.read_excel("bus.xlsx",sheet_name='data')
x = data_test.values[:,0:]
x


# In[289]:


def convert(): 
    newArr=[]
    for item in x:
        newArr.append(LatLongToMerc(item))
    newArr = np.array(newArr)
    return newArr
convert()


# In[161]:


import math
 
def LatLongToMerc(arr):
    lat = arr[1]
    lon = arr[0]
    if lat>89.5:
        lat=89.5
    if lat<-89.5:
        lat=-89.5
 
    rLat = math.radians(lat)
    rLong = math.radians(lon)
 
    a=6378137.0 #a - большая полуось эллипса;
    b=6356752.3142 #b - малая полуось эллипса;
    f=(a-b)/a
    e=math.sqrt(2*f-f**2) #эксцентриситет эллипса
    x=a*rLong
    y=a*math.log(math.tan(math.pi/4+rLat/2)*((1-e*math.sin(rLat))/(1+e*math.sin(rLat)))**(e/2))
    return [round(x,4),round(y,4)]
 


# In[309]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
def clastering(newArr):
    k_means = KMeans(init = "k-means++",n_clusters = 10,n_init=10)
    k_means.fit(newArr)
    center  = k_means.cluster_centers_
    labels = k_means.labels_
    c = Counter(labels)
    c["center"] = center
    return c
clastering(convert())


# In[278]:





# In[339]:


import matplotlib.pyplot as plt
def drow(obj,newArr):
    center = obj["center"]
    fig,ax =plt.subplots()
    plt.figure(figsize=(14,8), dpi = 80,facecolor="w",edgecolor="k")
    ax.plot(center[:,0],center[:,1], 'ro',markersize=10,label="Кластер")
    ax.plot(newArr[:,0],newArr[:,1], 'gs',markersize=1,label = "Остановка")
    plt.gca().set(xlim = (6629000,6720500),ylim = (3340500,3660000),xlabel = "x,м.",ylabel = "y,м.")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend()
    plt.plot(center[:,0],center[:,1], 'ro',markersize=10,label="Кластер")
    plt.plot(newArr[:,0],newArr[:,1], 'gs',markersize=1,label = "Остановка")
    plt.title("Остановки Санкт-Петербурга")
    plt.show()
data = convert()
drow(clastering(data),data)


# In[ ]:




