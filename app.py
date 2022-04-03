import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, json, jsonify

app = Flask(__name__)

def deserialize_and_load(filename):
    with open(f'model/{filename}', 'rb') as f:
        obj = pickle.load(f)
    return obj

def upload_strong_feats():
    with open('model/strong_feats.json', 'r') as f:
        strong_feats = json.load(f)
    return strong_feats

def change_column_types(df, strong_feats):
    df[strong_feats['num']] = df[strong_feats['num']].astype(float)
    df[strong_feats['cat']] = df[strong_feats['cat']].astype(str)
    df[strong_feats['cat']] = df[strong_feats['cat']].replace('nan', '#nan')
    return df

def transform(transformer, X):
    X_transformed = transformer.transform(X)
    X = pd.DataFrame(X_transformed, columns = X.columns)
    return X

def combine_num_and_cat(X_num, X_cat):
    X = pd.concat([X_num, X_cat], axis = 1)
    return X

# upload model and aditional transformers
model = deserialize_and_load('model.pkl')
imp = deserialize_and_load('imputer.pkl')
enc = deserialize_and_load('encoder.pkl')
feats = upload_strong_feats()


@app.route('/')
def home():
    return "PD Inference"
 
@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    df = pd.DataFrame(data)
    df = change_column_types(df, feats)
    x_num = transform(imp, df[feats['num']])
    x_cat = transform(enc, df[feats['cat']])
    x = combine_num_and_cat(x_num, x_cat)   
    preds = model.predict_proba(x)[:, 1]
    output = {'prediction': list(np.around(preds, 5))}
    return jsonify(output)


if __name__ == '__main__':
    app.run()