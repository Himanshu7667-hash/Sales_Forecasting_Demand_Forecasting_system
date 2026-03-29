from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import generate_inventory_report, get_low_stock_products, get_near_expiry_products

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'data_set'
app.config['MODEL_PATH'] = 'trained_model.pkl'
app.config['DATA_PATH'] = 'data_set/data.csv'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load the pickled model
def load_trained_model():
    """Load the trained model with proper error handling"""
    try:
        with open(app.config['MODEL_PATH'], 'rb') as model_file:
            model_data = pickle.load(model_file)
        
        # Check if the loaded data is a model or a dictionary containing a model
        if hasattr(model_data, 'predict'):
            print("Model loaded successfully!")
            return model_data
        elif isinstance(model_data, dict) and 'model' in model_data:
            print("Model loaded successfully from dictionary!")
            return model_data['model']
        else:
            print("Loaded data is not a valid model")
            return None
    except FileNotFoundError:
        print(f"Model file not found at {app.config['MODEL_PATH']}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def simple_prediction(quantity1, quantity2, quantity3):
    """Simple prediction using weighted average"""
    try:
        # Simple weighted average prediction
        weights = [0.2, 0.3, 0.5]  # Give more weight to recent data
        prediction = (quantity1 * weights[0] + quantity2 * weights[1] + quantity3 * weights[2])
        return prediction
    except Exception as e:
        print(f"Simple prediction error: {str(e)}")
        return None

model = load_trained_model()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with improved error handling"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file part"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No selected file"
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "Please upload a CSV file"
            }), 400
        
        # Save the file to the data directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
        file.save(file_path)
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully!"
        }), 200
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error saving file: {str(e)}"
        }), 500

@app.route('/inventory')
def inventory():
    """Display inventory with restocking and expiry recommendations"""
    try:
        # Check if data file exists
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', 
                                 error="Data file not found. Please upload a CSV file first.")
        
        # Read data from CSV file
        df = pd.read_csv(app.config['DATA_PATH'])
        
        # Get recommendations for restocking and near expiry products
        low_stock_recommendations = get_low_stock_products(df)
        near_expiry_recommendations = get_near_expiry_products(df)
        
        # Get inventory metrics
        from utils import calculate_inventory_metrics
        metrics = calculate_inventory_metrics(df)
        
        return render_template('inventory.html', 
                             restock_recommendations=low_stock_recommendations,
                             near_expiry_recommendations=near_expiry_recommendations,
                             metrics=metrics)
    except Exception as e:
        return render_template('error.html', error=f"Error loading inventory: {str(e)}")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    """Handle prediction requests with improved error handling"""
    if request.method == "POST":
        try:
            # Extract input data from the request
            data = request.get_json()
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No data provided"
                }), 400
            
            quantity1 = float(data.get('quantity1', 0))
            quantity2 = float(data.get('quantity2', 0))
            quantity3 = float(data.get('quantity3', 0))

            # Validate input data
            if quantity1 < 0 or quantity2 < 0 or quantity3 < 0:
                return jsonify({
                    "success": False,
                    "error": "Quantities must be non-negative"
                }), 400

            # Try to use the trained model first
            if model is not None and hasattr(model, 'predict'):
                try:
                    # Prepare the input data for prediction
                    input_data = np.array([[quantity1, quantity2, quantity3]])
                    prediction_value = model.predict(input_data)[0][0]
                    return jsonify({
                        "success": True,
                        "prediction": float(prediction_value)
                    })
                except Exception as model_error:
                    print(f"Model prediction failed: {str(model_error)}")
                    # Fall back to simple prediction
                    pass

            # Fallback to simple prediction
            prediction_value = simple_prediction(quantity1, quantity2, quantity3)
            if prediction_value is not None:
                return jsonify({
                    "success": True,
                    "prediction": float(prediction_value),
                    "method": "simple_weighted_average"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to make prediction"
                }), 500

        except ValueError as e:
            return jsonify({
                "success": False,
                "error": f"Invalid input data: {str(e)}"
            }), 400
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Failed to make prediction: {str(e)}"
            }), 500

    elif request.method == "GET":
        return render_template("prediction.html")

@app.route('/analytics')
def sales_analytics():
    """Display sales analytics with improved error handling"""
    try:
        # Check if data file exists
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', 
                                 error="Data file not found. Please upload a CSV file first.")
        
        # Load data from CSV file
        data = pd.read_csv(app.config['DATA_PATH'])
        
        # Calculate total sales and average order value
        total_sales = float(data["total_revenue"].sum())
        average_order_value = float(data["total_revenue"].mean())

        # Find top 5 selling and bottom 5 selling products based on quantity_stock
        # Since we don't have quantity_sold, we'll use quantity_stock as a proxy
        top_selling_products = data.nlargest(5, "quantity_stock")
        bottom_selling_products = data.nsmallest(5, "quantity_stock")

        # Convert DataFrames to dictionaries and ensure JSON serializable
        top_selling_dict = []
        for _, row in top_selling_products.iterrows():
            top_selling_dict.append({
                'product_id': int(row['product_id']),
                'product_name': str(row['product_name']),
                'quantity_stock': int(row['quantity_stock']),
                'total_revenue': float(row['total_revenue'])
            })

        bottom_selling_dict = []
        for _, row in bottom_selling_products.iterrows():
            bottom_selling_dict.append({
                'product_id': int(row['product_id']),
                'product_name': str(row['product_name']),
                'quantity_stock': int(row['quantity_stock']),
                'total_revenue': float(row['total_revenue'])
            })

        # Create sales trend plot
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(data)), data['total_revenue'], marker='o', linestyle='-', linewidth=2, markersize=4)
            ax.set_title('Revenue Trend by Product', fontsize=16, fontweight='bold')
            ax.set_xlabel('Product Index', fontsize=12)
            ax.set_ylabel('Total Revenue', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            fig.tight_layout()

            # Save the plot to a static file
            sales_trend_file_path = "static/sales_trend.png"
            fig.savefig(sales_trend_file_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        except Exception as e:
            print(f"Error creating sales trend plot: {e}")

        return render_template('analytics.html', 
                             total_sales=total_sales,
                             average_order_value=average_order_value,
                             top_selling_products=top_selling_dict,
                             bottom_selling_products=bottom_selling_dict)
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return render_template('error.html', error=f"Error loading analytics: {str(e)}")

@app.route('/train', methods=['POST'])
def train_model():
    """Train the prediction model"""
    try:
        from Prediction import main as train_prediction_model
        print("Starting model training process...")
        success = train_prediction_model()
        
        if success:
            # Reload the model
            global model
            model = load_trained_model()
            if model is not None:
                return jsonify({
                    "success": True,
                    "message": "Model trained successfully!"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Model training completed but failed to load the model."
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": "Model training failed. Check the logs for details."
            }), 500
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error training model: {str(e)}"
        }), 500

@app.route('/api/inventory-summary')
def inventory_summary():
    """API endpoint for inventory summary"""
    try:
        print(f"Checking for data file at: {app.config['DATA_PATH']}")
        
        if not os.path.exists(app.config['DATA_PATH']):
            print("Data file not found, returning default metrics")
            return jsonify({
                "metrics": {
                    "total_products": 0,
                    "low_stock_count": 0,
                    "average_stock_level": 0,
                    "total_stock_value": 0,
                    "near_expiry_count": 0,
                    "total_revenue": 0,
                    "average_order_value": 0
                },
                "message": "No data file found. Please upload a CSV file first."
            }), 200
        
        print("Data file found, generating report...")
        report = generate_inventory_report(app.config['DATA_PATH'])
        
        if report:
            print(f"Report generated successfully: {report}")
            return jsonify(report)
        else:
            print("Failed to generate report")
            return jsonify({
                "metrics": {
                    "total_products": 0,
                    "low_stock_count": 0,
                    "average_stock_level": 0,
                    "total_stock_value": 0,
                    "near_expiry_count": 0,
                    "total_revenue": 0,
                    "average_order_value": 0
                },
                "error": "Failed to generate report"
            }), 200
    except Exception as e:
        print(f"Error in inventory summary: {str(e)}")
        return jsonify({
            "metrics": {
                "total_products": 0,
                "low_stock_count": 0,
                "average_stock_level": 0,
                "total_stock_value": 0,
                "near_expiry_count": 0,
                "total_revenue": 0,
                "average_order_value": 0
            },
            "error": str(e)
        }), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
    
