import numpy as np
from flask import Flask, request, render_template
import pickle

#create an app object using the Flask class
app= Flask(__name__)

#load the trained model using pickle
model=pickle.load(open('model/heart_failure_prediction_model.pkl', 'rb'))

#define the route to be home
#running the app sends us to index.html
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        chestpaintype = int(request.form.get('chestpaintype'))
        restingbp = int(request.form.get('restingbp'))
        cholesterol = int(request.form.get('cholesterol'))
        fastingbs = int(request.form.get('fastingbs'))
        restingecg = int(request.form.get('restingecg'))
        maxhr = int(request.form.get('maxhr'))
        exerciseangina = int(request.form.get('exerciseangina'))
        oldpeak = float(request.form.get('oldpeak'))
        stslope = int(request.form.get('stslope'))

        features=[[age, restingbp, cholesterol, fastingbs, maxhr, oldpeak, sex, chestpaintype, restingecg, exerciseangina, stslope]]      

        prediction=model.predict(features)

        if prediction==[0]:
            status="Negative"
        else:
            status="Positive"

        return render_template('predict.html', response='Your are {}'.format(status))

if __name__ == "__main__":
    print("Starting Python Flask Server For heart Failure Prediction...")
    app.run(debug=True)