from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
import joblib
import h5py

app = Flask(__name__)
CORS(app) # Allows frontend to communicate with backend

def load_model_with_compat(path):
    """Load model and patch newer InputLayer config keys if needed."""
    try:
        return tf.keras.models.load_model(path)
    except TypeError as e:
        print(f"Primary model load failed, retrying with compatibility patch: {e}")

        with h5py.File(path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            model_config = json.loads(model_config)

        def patch_input_layers(node):
            if isinstance(node, dict):
                if node.get('class_name') == 'InputLayer' and isinstance(node.get('config'), dict):
                    cfg = node['config']
                    if 'batch_shape' in cfg and 'batch_input_shape' not in cfg:
                        cfg['batch_input_shape'] = cfg.pop('batch_shape')
                    cfg.pop('optional', None)
                if isinstance(node.get('config'), dict):
                    # Keras 3 key unsupported by TF/Keras 2.15 deserializer.
                    node['config'].pop('quantization_config', None)
                    dtype_cfg = node['config'].get('dtype')
                    if isinstance(dtype_cfg, dict):
                        dtype_name = dtype_cfg.get('config', {}).get('name')
                        if dtype_name:
                            node['config']['dtype'] = dtype_name
                for value in node.values():
                    patch_input_layers(value)
            elif isinstance(node, list):
                for item in node:
                    patch_input_layers(item)

        patch_input_layers(model_config)

        model = tf.keras.models.model_from_json(json.dumps(model_config))
        model.load_weights(path)
        return model

# 1. Load the exported assets from Colab
print("Loading NNDL Model & Data...")
model = load_model_with_compat('nndl_churn_model.h5')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('model_features.pkl')
df = pd.read_csv('bank_customers_data.csv')

# Pre-calculate predictions for speed
X_data = df.drop(columns=['customerId', 'bankId', 'name', 'bankName', 'managerId', 'managerName', 'Churn'], errors='ignore')
X_data = pd.get_dummies(X_data, drop_first=True)

# Ensure columns match training exactly
X_data = X_data.reindex(columns=feature_cols, fill_value=0) 

X_scaled = scaler.transform(X_data)
df['Churn_Prob'] = model.predict(X_scaled).flatten() * 100

@app.route('/api/portal', methods=['GET'])
def get_portal():
    banks = df[['bankId', 'bankName', 'managerName']].drop_duplicates().to_dict('records')
    return jsonify({'banks': banks})

@app.route('/api/bank/<bank_id>', methods=['GET'])
def get_bank_data(bank_id):
    bank_df = df[df['bankId'] == bank_id]
    
    total = len(bank_df)
    at_risk = len(bank_df[bank_df['Churn_Prob'] > 60])
    
    top_risk = bank_df.sort_values(by='Churn_Prob', ascending=False).head(5)
    
    return jsonify({
        'bank_name': str(bank_df['bankName'].iloc[0]),
        'manager': str(bank_df['managerName'].iloc[0]),
        'total': int(total),
        'at_risk': int(at_risk),
        'safe': int(total - at_risk),
        'top_risk': top_risk[['customerId', 'name', 'Churn_Prob']].to_dict('records'),
        'all_customers': bank_df[['customerId', 'name', 'tenure', 'monthlyCharges', 'Churn_Prob']].to_dict('records')
    })

@app.route('/api/analyze/<cust_id>', methods=['GET'])
def analyze_customer(cust_id):
    # Find the specific customer
    cust = df[df['customerId'] == cust_id].iloc[0]
    prob = cust['Churn_Prob']
    
    # 🟢 THE FIX: Explicitly wrap the Pandas/NumPy data in standard Python int(), float(), and str()
    return jsonify({
        'id': str(cust['customerId']),
        'name': str(cust['name']),
        'contract': str(cust['contractType']),
        'billing': float(cust['monthlyCharges']),
        'tenure': int(cust['tenure']),
        'calls': int(cust['supportCalls']),
        'prob': float(round(prob, 1))
    })


from flask import send_from_directory

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
