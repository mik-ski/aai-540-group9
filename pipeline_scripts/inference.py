import json
import os
import io
import numpy as np
import lightgbm as lgb

def model_fn(model_dir):
    """
    Load the LightGBM model from the model directory.
    This function is called by SageMaker when the endpoint starts.
    """
    model_file = os.path.join(model_dir, 'lgb_model.txt')
    metadata_file = os.path.join(model_dir, 'model_metadata.json')
    
    model = lgb.Booster(model_file=model_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return {'model': model, 'metadata': metadata}

def input_fn(input_data, content_type='application/json'):
    """
    Deserialize input data. Handles JSON and numpy formats.
    """
    if content_type == 'application/json':
        data = json.loads(input_data)
        # Handle both direct list and dict with 'features' key
        if isinstance(data, dict) and 'features' in data:
            return np.array(data['features'])
        return np.array(data)
    elif content_type == 'application/x-npy':
        # Handle numpy binary format (default SKLearn serializer)
        return np.load(io.BytesIO(input_data), allow_pickle=False)
    raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, model_dict):
    """
    Run prediction on input data.
    """
    model = model_dict['model']
    metadata = model_dict['metadata']
    threshold = metadata['threshold']
    
    # Ensure input is numpy array
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    
    # Reshape if needed (single sample)
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    # Get predictions
    probabilities = model.predict(input_data)
    predictions = (probabilities >= threshold).astype(int)
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'threshold': threshold,
        'feature_names': metadata.get('feature_cols', [])
    }

def output_fn(prediction, accept='application/json'):
    """
    Serialize prediction output.
    """
    return json.dumps(prediction), 'application/json'
