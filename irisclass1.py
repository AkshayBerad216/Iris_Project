from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('log_class_model.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home1.html')

@app.route('/predict', methods=['POST'])
def home():
    SepalLengthCm = request.form['SepalLengthCm']
    SepalWidthCm = request.form['SepalWidthCm']
    PetalLengthCm = request.form['PetalLengthCm']
    PetalWidthCm = request.form['PetalWidthCm']

    arr = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    pred = model.predict(arr)
    print('SepalLengthCm == ',SepalLengthCm)
    print('SepalWidthCm == ',SepalWidthCm)
    print("PetalLengthCm == ",PetalLengthCm)
    print("PetalWidthCm == ",PetalWidthCm)
    print("Prediction == ",pred)
    return render_template('after1.html', data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)   
