# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 08:57:16 2021

@author: Naushina Farheen S
"""

import pandas as pd # deals with data frame 
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 
pip install dtale --user
import dtale as dt

adult = pd.read_csv("C:/Users/Naushina Farheen S/Dropbox/My PC (DESKTOP-4RG72G3)/Desktop/NAUSHINA FOLDER/Practise Folder/adult.csv") # For Loading csv data.

for col in adult.columns:
    print(col)

adult = adult.rename({'education-num': 'educationnum', 'marital-status': 'maritalstatus', 'capital-gain': 'cg', 'capital-loss': 'cl', 'hours-per-week': 'hpw'}, axis=1)

adult = pd.DataFrame(adult)
adult.isnull().sum()

adult.describe()

adult.var()
adult.skew()

import dtale
dtale.show(adult)
d.open_browser()


#### Dtale
d=dt.show(adult)
d.open_browser()

adult['occupation'].unique()
adult['country'].unique()

adult.mode()

plt.hist(adult["output variable"])  #right skewed =0.647
plt.hist(XYZ["input variable"])  #right skewed =0.858
plt.boxplot(XYZ["output variable"],0,"rs",0) #Graphical representation for outliers present
plt.boxplot(XYZ["input variable"],0,"rs",0) 
#plt.bar(output = XYZ, x = np.arange(1, 110, 1))
plt.hist(XYZ) #histogram
plt.boxplot(XYZ) #boxplot

