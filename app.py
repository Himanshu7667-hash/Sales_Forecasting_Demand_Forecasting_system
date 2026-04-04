import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import warnings
from functools import lru_cache
from threading import Thread

from utils import generate_inventory_report, get_low_stock_products, get_near_expiry_products

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Config
app.config['UPLOAD_FOLDER'] = 'data_set'
app.config['MODEL_PATH'] = 'trained_model.pkl'
app.config['DATA_PATH'] = 'data_set/data.csv'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# ✅ Load model once
def load_model():
    try:
        with open(app.config['MODEL_PATH'], 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

# ✅ Cache dataset (MAIN FIX)
@lru_cache(maxsize=1)
def load_data():
    try:
        df = pd.read_csv(app.config['DATA_PATH'])

        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if 'minimum_stock_level' not in df.columns:
            df['minimum_stock_level'] = 10

        if 'quantity_stock' not in df.columns:
            df['quantity_stock'] = np.random.randint(10, 100, len(df))

        return df.head(5000)
    except:
        return pd.DataFrame()

# Home
@app.route('/')
def home():
    return render_template("index.html")

# Upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
    file.save(path)

    load_data.cache_clear()

    return jsonify({"success": True})

# Inventory
@app.route('/inventory')
def inventory():
    df = load_data()

    low_stock = get_low_stock_products(df)
    near_expiry = get_near_expiry_products(df)

    from utils import calculate_inventory_metrics
    metrics = calculate_inventory_metrics(df)

    return render_template('inventory.html',
                           restock_recommendations=low_stock,
                           near_expiry_recommendations=near_expiry,
                           metrics=metrics)

# Prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        q1 = float(data.get('quantity1', 0))
        q2 = float(data.get('quantity2', 0))
        q3 = float(data.get('quantity3', 0))

        if model:
            try:
                pred = model.predict([[q1, q2, q3]])[0][0]
                return jsonify({"prediction": float(pred)})
            except:
                pass

        # fallback
        pred = q1*0.2 + q2*0.3 + q3*0.5
        return jsonify({"prediction": pred})

    return render_template("prediction.html")

# Analytics
@app.route('/analytics')
def analytics():
    data = load_data()

    total_sales = float(data["total_revenue"].sum())
    avg_order = float(data["total_revenue"].mean())

    top = data.nlargest(5, "quantity_stock")
    bottom = data.nsmallest(5, "quantity_stock")

    sample = data.head(1000)

    plt.figure()
    plt.plot(sample['total_revenue'])
    plt.savefig("static/sales_trend.png")
    plt.close()

    return render_template('analytics.html',
                           total_sales=total_sales,
                           average_order_value=avg_order,
                           top_selling_products=top.to_dict('records'),
                           bottom_selling_products=bottom.to_dict('records'))

# Async training
@app.route('/train', methods=['POST'])
def train():
    from Prediction import main

    Thread(target=main).start()

    return jsonify({"message": "Training started"})

# API
@app.route('/api/inventory-summary')
def summary():
    df = load_data()
    report = generate_inventory_report(df)
    return jsonify(report)

# Run
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True) 
