import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.metrics import mean_absolute_error

BASE_DIR        = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNHEALTHY_DIR   = os.path.join(BASE_DIR, 'data/unhelathy-dataset/Unhealthy Normalized')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR, 'data/multi_platform_cpgs.pkl')
BASELINES_DIR   = os.path.join(BASE_DIR, 'baselines')
ALTUMAGE_H5     = os.path.join(BASELINES_DIR, 'AltumAge.h5')
ALTUMAGE_SCALER = os.path.join(BASELINES_DIR, 'scaler.pkl')
PRED_CSV        = os.path.join(BASELINES_DIR, 'unhealthy_predictions.csv')
ACCEL_CSV       = os.path.join(BASELINES_DIR, 'unhealthy_acceleration.csv')

DISEASE_MAP = {
    'GSE19711':    'Ovarian Cancer',
    'GSE41037':    'Schizophrenia',
    'GSE27044':    'Osteoporosis',
    'GSE99624':    'Osteoporosis',
    'E-GEOD-44763':'Osteoporosis',
    'GSE77241':    'Osteoporosis',
}

print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
print(f'  {len(AltumAge_cpgs)} CpG sites')

print('\nLoading unhealthy dataset...')

unhealthy_frames = []
for filename in os.listdir(UNHEALTHY_DIR):
    if filename.endswith('.pkl'):
        df = pd.read_pickle(os.path.join(UNHEALTHY_DIR, filename))
        cols = [c for c in df.columns if c in AltumAge_cpgs]
        cols += ['age', 'gender', 'dataset', 'tissue_type']
        df = df[[c for c in cols if c in df.columns]]
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age'])
        if len(df) > 0:
            unhealthy_frames.append(df)

unhealthy_data = pd.concat(unhealthy_frames)
unhealthy_data['disease'] = unhealthy_data['dataset'].map(DISEASE_MAP).fillna('Unknown')

X_unhealthy = unhealthy_data.drop(
    columns=['age', 'gender', 'dataset', 'tissue_type', 'disease'],
    errors='ignore').astype('float')
y_unhealthy = unhealthy_data['age'].astype('float')

print(f'  Unhealthy samples: {len(unhealthy_data)}')
print(unhealthy_data['disease'].value_counts())

print('\n--- AltumAge (2022) ---')
print(f'  TensorFlow version: {tf.__version__}')

altum_model = tf.keras.models.load_model(ALTUMAGE_H5, compile=False)
print(f'  Model loaded: input={altum_model.input_shape}')

with open(ALTUMAGE_SCALER, 'rb') as f:
    altum_scaler = pickle.load(f)

X_altum_scaled = altum_scaler.transform(
    X_unhealthy[AltumAge_cpgs]).astype(np.float32)

pred_altumage = altum_model(
    X_altum_scaled, training=False).numpy().flatten()

mae = mean_absolute_error(y_unhealthy, pred_altumage)
print(f'  MAE={mae:.3f}')
print(f'  Mean age acceleration: {(pred_altumage - y_unhealthy.values).mean():.3f}')

print('\nMerging into existing predictions CSV...')

if os.path.exists(PRED_CSV):
    existing = pd.read_csv(PRED_CSV)
    existing['AltumAge'] = pred_altumage
    existing['AltumAge_accel'] = pred_altumage - existing['true_age']
    existing.to_csv(PRED_CSV, index=False)
    print(f'  Updated: {PRED_CSV}')
else:
    result_df = pd.DataFrame({
        'true_age':      y_unhealthy.values,
        'dataset':       unhealthy_data['dataset'].values,
        'disease':       unhealthy_data['disease'].values,
        'AltumAge':      pred_altumage,
        'AltumAge_accel': pred_altumage - y_unhealthy.values,
    })
    result_df.to_csv(PRED_CSV, index=False)
    print(f'  Created: {PRED_CSV}')

print('\nUpdating acceleration summary...')

if os.path.exists(ACCEL_CSV):
    summary = pd.read_csv(ACCEL_CSV)
    for disease in unhealthy_data['disease'].unique():
        mask = unhealthy_data['disease'] == disease
        accel = pred_altumage[mask.values] - y_unhealthy.values[mask.values]
        idx   = summary[summary['disease'] == disease].index
        if len(idx) > 0:
            summary.loc[idx, 'AltumAge_mean_accel'] = round(float(accel.mean()), 3)
            summary.loc[idx, 'AltumAge_std_accel']  = round(float(accel.std()),  3)
    summary.to_csv(ACCEL_CSV, index=False)
    print(f'  Updated: {ACCEL_CSV}')

print('\n' + '='*60)
print('ALTUMAGE AGE ACCELERATION BY DISEASE')
print('='*60)
for disease in ['Ovarian Cancer', 'Schizophrenia', 'Osteoporosis']:
    mask  = unhealthy_data['disease'] == disease
    accel = pred_altumage[mask.values] - y_unhealthy.values[mask.values]
    n     = mask.sum()
    print(f'{disease:20s}  n={n:3d}  '
          f'mean={accel.mean():+.3f}  std={accel.std():.3f}')
