from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


# Load the KNN model
model_path = "knn_model.pkl"
with open(model_path, 'rb') as file:
    knn_model = pickle.load(file)


@app.route("/")
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form['sex']
        job = (request.form['job'])
        housing = request.form['housing']
        savings = request.form['savingAccount']
        checking = request.form['checkingAccount']
        credit = (request.form['creditAmount'])
        duration = (request.form['duration'])
        purpose = request.form['purpose']

    input_features = np.array([[age, sex, job, housing, savings, checking, credit, duration, purpose]])
    prediction = knn_model.predict(input_features)
    print(prediction)
    
    if prediction[0] == 1:
        result = "Good"
    else: 
        result = "Bad"

    return render_template('result.html', prediction=result)

    
    



if __name__ == '__main__':
    app.run(debug=True)