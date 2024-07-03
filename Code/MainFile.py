#============================= IMPORT LIBRARIES =============================

import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import base64


#  ---------- BACKGROUND IMAGE

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')



st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Prediction of Electricity Bill using SML Technique"}</h1>', unsafe_allow_html=True)


#============================= DATA SELECTION ==============================

dataframe=pd.read_csv("Household energy bill data.csv")

print("----------------------------------------------------")
print("Input Data          ")
print("----------------------------------------------------")
print()
print(dataframe.head(20))
print()

#============================= PREPROCESSING ==============================

#==== CHECKING MISSING VALUES ====

print("----------------------------------------------------")
print("Before checking Missing Values          ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())
print()

res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()       
    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        
        print(dataframe.isnull().sum())
        

# ======================= EDA Analysis  

# --- BIVARIENT ANALYSIS

import matplotlib.pyplot as plt 
import seaborn as sns
plt.figure(figsize=(6, 5)) 
sns.barplot(x=dataframe['num_rooms'], y=dataframe['num_people']) 
plt.title("Number of rooms vs Number of People")
plt.show()

# --- SCATTER PLOT

sns.scatterplot(x=dataframe['num_rooms'], y=dataframe['num_people']) 
plt.title("Scatter Plot")
plt.show()


data_ac = dataframe['is_ac']
data_tv = dataframe['is_tv']

data_flat = dataframe['is_flat']

data_urban = dataframe['is_urban']


#============================= DATA SPLITTING ==============================

print("----------------------------------------------------")
print("Data Splitting          ")
print("----------------------------------------------------")
print()

from sklearn.model_selection import train_test_split

X = dataframe.drop('amount_paid', axis=1)
y = dataframe['amount_paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

print("Total no of data's       :",dataframe.shape[0])
print()
print("Total no of Train data's :",X_train.shape[0])
print()
print("Total no of Test data's  :",X_test.shape[0])
print()
print()




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtr = scaler.fit_transform(X_train)

xt = scaler.transform(X_test)







#============================= CLASSIFICATION ==============================

# ==== LINEAR REGRESSION ====

from sklearn import linear_model

lasso_r=linear_model.LinearRegression()

lasso_r.fit(xtr,y_train)

pred_lasso=lasso_r.predict(xt)

from sklearn import metrics
mae_lr=metrics.mean_absolute_error(pred_lasso,y_test)

mse_lr=metrics.mean_squared_error(pred_lasso,y_test)
                                        
import math
rsme_lr = math.sqrt(mse_lr)

print("---------------------------------------")
print("Machine Learning ----> Linear Regression")
print("---------------------------------------")
print()
print("==================================================")
print("1. Mean Absolute Error     :",mae_lr)
print()
print("2. Mean Squared Error      :",mse_lr)
print()
print("3. Root Mean Squared Error :",rsme_lr)
print("==================================================")
print()

# ==== RIDGE REGRESSION ====

ridge_r=linear_model.Ridge()

ridge_r.fit(xtr,y_train)

pred_ridge=ridge_r.predict(xt)

mae_rr=metrics.mean_absolute_error(pred_ridge,y_test)

mse_rr=metrics.mean_squared_error(pred_ridge,y_test)
                                        
import math
rsme_rr = math.sqrt(mse_rr)

print("---------------------------------------")
print("Machine Learning ----> Ridge Regression")
print("---------------------------------------")
print()
print("==================================================")
print("1. Mean Absolute Error     :",mae_rr)
print()
print("2. Mean Squared Error      :",mse_rr)
print()
print("3. Root Mean Squared Error :",rsme_rr)
print("==================================================")
print()



#============================= PREDICTION ==============================

# === ELECTRICITY PRICE ===

print("---------------------------------------")
print("Prediction ---> Electricity Price")
print("---------------------------------------")
print()


for i in range(0,10):
    print("-------------------------------")
    print()
    print([i],"The Electricity price =",pred_ridge[i])    


print()
print("---------------------------------------------------------------")
print()

import matplotlib.pyplot as plt

plt.title("Predicting Electricity Price - Linear" )
plt.plot(pred_lasso) 
plt.show() 

print()
print("---------------------------------------------------------------")
print()

plt.title("Predicting Electricity Price - Ridge" )
plt.plot(pred_ridge) 
plt.show() 


print()
print("------------------------------------------------------------")
print() 


# === COMPARISON ===

import matplotlib.pyplot as plt
import numpy as np


objects = ('Linear Regression', 'Ridge Regression')
y_pos = np.arange(len(objects))
performance = [mae_lr,mae_rr]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance ')
plt.title('Comparison Graph -- Error Values')
plt.show()



#########################################################


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:28px;">{" Kindly fill the below details"}</h1>', unsafe_allow_html=True)


a1 = st.text_input("Enter Number of Rooms ",0)

a2 = st.text_input("Enter Number of Peoples ",0)

a3 = st.text_input("Enter House Area ",0)

a4= st.selectbox("Is AC",data_ac)

a5= st.selectbox("Is TV",data_tv)

a6= st.selectbox("Is Flat",data_flat)

a7= st.text_input("Enter Average Monthly Income",0)

a8= st.text_input("Enter Number of Children ",0)

a9= st.selectbox("Is Urban",data_urban)


butt=st.button("Predict")

if butt:


    Data_reg = [int(a1),int(a2),int(a3),int(a4),int(a5),int(a6),int(a7),int(a8),int(a9)]
    # st.write(Data_reg)
    
    y_pred_reg=ridge_r.predict([Data_reg])
    
    res = " Generated Electricity Bill  = " + str(y_pred_reg)
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:28px;">{res}</h1>', unsafe_allow_html=True)
    







