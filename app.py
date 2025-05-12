from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle de prédiction d'AVC avec pickle
with open('besthealthcare1.pkl', 'rb') as f:
    model = pickle.load(f)

# Encodage des variables catégorielles
gender_map = {'Male': 1, 'Female': 0}
ever_married_map = {'Yes': 1, 'No': 0}
work_type_map = {
    'Private': 0,
    'Self-employed': 1,
    'Govt_job': 2,
    'children': 3,
    'Never_worked': 4
}
residence_type_map = {'Urban': 1, 'Rural': 0}
smoking_status_map = {
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = [
        gender_map[request.form['gender']],
        float(request.form['age']),
        int(request.form['hypertension']),
        int(request.form['heart_disease']),
        ever_married_map[request.form['ever_married']],
        work_type_map[request.form['work_type']],
        residence_type_map[request.form['Residence_type']],
        float(request.form['avg_glucose_level']),
        float(request.form['bmi']),
        smoking_status_map[request.form['smoking_status']]
    ]

    features = np.array([data])
    prediction = model.predict(features)[0]
    result = ' Risque d’AVC détecté' if prediction == 1 else ' Aucun risque détecté'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
