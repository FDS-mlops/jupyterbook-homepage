#!/usr/bin/env python
#_*_ coding:utf-8 _*_

# Core pkgs
from io import BytesIO
from io import StringIO
import os
import sys
import streamlit as st
from streamlit import caching
caching.clear_cache()

# EDA pkgs
import pandas as pd
import numpy as np

# Viz pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# ML pgks
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

FILE_TYPES = ["csv", "txt"]

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():

    st.title("JupyterBook Auto ML App")
    st.text("So Simple, it makes you happier!")

    activities = ["EDA", "Plot", "Model Building", "About"]

    choice = st.sidebar.selectbox("Select Activity", activities)
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload Dataset", type=FILE_TYPES)    
        if data is not None:
            df = pd.read_csv(data, encoding="utf-8")       
            st.dataframe(df.head())

            if st.checkbox("Show shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Show Columns To Show"):
                selected_columns = st.multiselect("Select Coluns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show Summary"):
                st.write(df.describe())
  
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())

            if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
            
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Select 1 Column", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
            

    elif choice == 'Plot':
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload Dataset", type=FILE_TYPES)    
        if data is not None:
            df = pd.read_csv(data, encoding="utf-8")       
            st.dataframe(df.head())

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", ["area","bar","line","hist","box","kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} ofr {}".format(type_of_plot,selected_columns_names))

                # Plot By Streamlit
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)
                
                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                # Custom Plot
                elif type_of_plot:
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

    elif choice == 'Model Building':
        st.subheader("Building ML Model")

        data = st.file_uploader("Upload Dataset", type=FILE_TYPES)
        st.text("Dataset Format: X = [:,0:-1], Y = [:,-1]")

        if data is not None:
            df = pd.read_csv(data, encoding="utf-8")       
            st.dataframe(df.head())

            # Preparation
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7

            # Model - evaluate each model
            models = []
            models.append(("LR",LogisticRegression()))
            models.append(("LDA",LinearDiscriminantAnalysis()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("CART",DecisionTreeClassifier()))
            models.append(("NB",GaussianNB()))
            models.append(("SVM",SVC()))

            # List
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'

            for name,model in models:
                kfold = model_selection.KFold(n_splits=10,random_state=seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"model_name":name, "model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
                all_models.append(accuracy_results)

            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Standard Deviation"]))

            if st.checkbox("Metrics as JSON"):
                st.json(all_models)

    elif choice == 'About':
        st.header("JupyterBook")
        st.write("JupyterBook is the company which providing the jupyter notebook server either on the public cloud and in On-premise.")

        st.write("We helps your data scientists to focus on more import matters.")
        
        st.write("Please contact us, if you want to build your own stable and scable data analytics platform powered by Kubernetes platform.")

        st.subheader("Online data platform service - PaaS")
        st.write("https://www.jupyteronline.com")

        st.subheader("Q&A")
        st.write("Email :    jupyteronline@gmail.com")  

if __name__ == '__main__':
    main()
