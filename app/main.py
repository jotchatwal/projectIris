from flask import Flask, request, jsonify, render_template
import pickle, os
import pandas as pd

app = Flask(__name__)

# ─── 1. LOAD YOUR IRIS PIPELINES ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model1 = pickle.load(open(os.path.join(BASE,'model1.pkl'),'rb'))  # Logistic
model2 = pickle.load(open(os.path.join(BASE,'model2.pkl'),'rb'))  # Decision Tree
model3 = pickle.load(open(os.path.join(BASE,'model3.pkl'),'rb'))  # Forest

models = {'lr': model1, 'dt': model2, 'rf': model3}
names  = {'lr': 'Logistic Regression', 'dt': 'Decision Tree', 'rf': 'Random Forest'}
species_map = {0:'setosa', 1:'versicolor', 2:'virginica'}

# ─── 2. FEATURE ENGINEERING FOR IRIS ─────────────────────────────────────────
def make_features_iris(sl, sw, pl, pw):
    # exactly the 7 columns your pipelines expect
    return pd.DataFrame([{
        'sepal length (cm)': sl,
        'sepal width (cm)' : sw,
        'petal length (cm)': pl,
        'petal width (cm)' : pw,
        'sepal_area': sl * sw,
        'petal_area': pl * pw,
        'sepal_petal_ratio': (sl / pl) if pl else 0
    }])

# ─── 3. BEAUTIFUL WEB FORM ───────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')   # see template below

@app.route('/predict', methods=['POST'])
def predict():
    key = request.form['model_name']
    sl = float(request.form['sepal_length'])
    sw = float(request.form['sepal_width'])
    pl = float(request.form['petal_length'])
    pw = float(request.form['petal_width'])

    X = make_features_iris(sl, sw, pl, pw)
    pred = models[key].predict(X)[0]

    return render_template('result.html',
      model_name=names[key],
      sl=sl, sw=sw, pl=pl, pw=pw,
      sa=X.at[0,'sepal_area'],
      pa=X.at[0,'petal_area'],
      spr=X.at[0,'sepal_petal_ratio'],
      pred=pred,
      species=species_map[pred]
    )

# ─── 4. JSON ENDPOINTS FOR POSTMAN ───────────────────────────────────────────
@app.route('/<key>/evaluate', methods=['POST'])
def evaluate(key):
    data = request.get_json(force=True)
    sl, sw, pl, pw = map(float, (
      data['sepal_length'],
      data['sepal_width'],
      data['petal_length'],
      data['petal_width']
    ))
    X = make_features_iris(sl, sw, pl, pw)
    pred = models[key].predict(X)[0]
    return jsonify({
      'prediction': int(pred),
      'species': species_map[pred]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
