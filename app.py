from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib

# تحميل النموذج والبيانات
model = joblib.load('random_forest_model.pkl')
meals_df = pd.read_csv('meals_data.csv')

# تحديد المسارات الصحيحة
app = Flask(__name__, template_folder='medi/templates', static_folder='medi/static')

@app.route('/')
def home():
    return render_template('index.html')  # الصفحة الرئيسية

@app.route('/services')
def services():
    return render_template('services.html')  # صفحة الإدخال

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    insulin = float(request.form['insulin'])

    meals = meals_df.copy()
    meals['age'] = age
    meals['bmi'] = bmi
    meals['insulin'] = insulin

    X = meals[['age', 'bmi', 'insulin', 'protein_(g)', 'carbohydrates_(g)', 'fat_(g)', 'calories_(kcal)']]
    meals['predicted_glucose'] = model.predict(X)

    top3 = meals.sort_values('predicted_glucose').head(3)['food_item'].tolist()

    return render_template('services.html', results=top3)

if __name__== '__main__':
    app.run(debug=True,port=5000)