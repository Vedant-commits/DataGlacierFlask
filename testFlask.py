from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('dun.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    return render_template("home.html", prediction_text=f'Predicted Iris Class: {prediction[0]}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    try: 
        from collections import Sequence
    except ImportError:
        from collections.abc import Sequence
    app.run(debug=True)
    