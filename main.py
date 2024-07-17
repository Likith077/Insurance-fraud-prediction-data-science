from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import LogisticRegression
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from csv import writer
import pandas as pd
from flask_material import Material

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home1():
    return render_template('home.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        age = request.form['age']
        policy_annual_premium = request.form['policy_annual_premium']
        policy_state = request.form['policy_state']
        insured_sex = request.form['insured_sex']
        insured_education_level = request.form['insured_education_level']
        incident_type = request.form['incident_type']
        collision_type = request.form['collision_type']
        incident_severity = request.form['incident_severity']
        number_of_vehicles_involved = request.form['number_of_vehicles_involved']
        witnesses = request.form['witnesses']
        injury_claim = request.form['injury_claim']
        property_claim = request.form['property_claim']
        Vehicle_Claim = request.form['Vehicle_Claim']

        sample_data = [age,policy_annual_premium,policy_state,insured_sex,insured_education_level,incident_type,collision_type,incident_severity,number_of_vehicles_involved,witnesses,injury_claim,property_claim,Vehicle_Claim]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)

        df = pd.read_csv('insurance_claims.csv')
        
        df = df.drop(columns = ['policy_number', 'insured_zip', 'policy_bind_date','incident_date', 'insured_occupation','incident_location', '_c39', 'auto_year', 'incident_hour_of_the_day','months_as_customer', 'policy_csl', 'policy_deductable', 'umbrella_limit', 'insured_hobbies','insured_relationship','capital-gains','capital-loss','incident_state', 'incident_city','police_report_available', 'total_claim_amount','bodily_injuries','property_damage','bodily_injuries','property_damage','authorities_contacted','auto_make', 'auto_model'])
        df['collision_type'].replace(to_replace='?', value=0, inplace=True)
        df=df.replace(['OH','IL','IN','MALE','FEMALE','JD','High School','Associate','MD','Masters','PhD','College','Multi-vehicle Collision','Single Vehicle Collision','Vehicle Theft','Parked Car','Rear Collision','Side Collision','Front Collision','0','Minor Damage','Total Loss','Major Damage','Trivial Damage','Y','N'],['0','1','2','0','1','0','1','2','3','4','5','6','0','1','2','3','0','1','2','3','0','1','2','3','0','1'])
        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234)

        DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
        DecisionTree.fit(X_train,y_train)

        result = DecisionTree.predict(ex1)
        if result=='0':
            class1="Fraud"
        else:
            class1="Not Fraud"

        return jsonify({'class1': class1})

if __name__ =='__main__':
    app.run(debug=True)