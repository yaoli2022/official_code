import pandas as pd
import numpy as np
import os

BASE_DIR      = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINES_DIR = os.path.join(BASE_DIR, 'baselines')
PRED_CSV      = os.path.join(BASELINES_DIR, 'unhealthy_predictions.csv')
OUTPUT_CSV    = os.path.join(BASELINES_DIR, 'unhealthy_age_group_acceleration.csv')

print('Loading predictions...')
df = pd.read_csv(PRED_CSV)
print(f'  Total samples: {len(df)}')
print(f'  Columns: {df.columns.tolist()}')

AGE_GROUPS = ['0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']

def age_group(age):
    if age <= 20:  return '0-20'
    if age <= 45:  return '20-45'
    if age <= 55:  return '45-55'
    if age <= 65:  return '55-65'
    if age <= 75:  return '65-75'
    if age <= 80:  return '75-80'
    return '80+'

df['age_group'] = df['true_age'].apply(age_group)

MODELS = ['Horvath', 'AltumAge', 'DeepMAge', 'ResnetAge',
          'PNA-GNN', 'PNA-GNN+CNN', 'PNA-GNN+Stat']

accel_cols = [f'{m}_accel' for m in MODELS if f'{m}_accel' in df.columns]
models_available = [c.replace('_accel', '') for c in accel_cols]
print(f'  Models available: {models_available}')

DISEASES = ['Ovarian Cancer', 'Schizophrenia', 'Osteoporosis']

all_rows = []

for disease in DISEASES:
    sub = df[df['disease'] == disease].copy()
    print(f'\n=== {disease} (n={len(sub)}) ===')

    grouped = sub.groupby('age_group')

    for age_grp in AGE_GROUPS:
        if age_grp not in grouped.groups:
            continue
        grp = grouped.get_group(age_grp)
        n   = len(grp)

        row = {
            'disease':   disease,
            'age_group': age_grp,
            'n':         n,
        }

        for model in models_available:
            col = f'{model}_accel'
            if col in grp.columns:
                row[f'{model}_mean_accel'] = round(float(grp[col].mean()), 3)
                row[f'{model}_std_accel']  = round(float(grp[col].std()),  3)
            else:
                row[f'{model}_mean_accel'] = np.nan
                row[f'{model}_std_accel']  = np.nan

        all_rows.append(row)

    result = sub.groupby('age_group')[accel_cols].mean().round(3)
    result.insert(0, 'n', sub.groupby('age_group').size())
    print(result.to_string())

result_df = pd.DataFrame(all_rows)
result_df.to_csv(OUTPUT_CSV, index=False)
print(f'\nSaved: {OUTPUT_CSV}')
print(f'Shape: {result_df.shape}')
print('\nFull table:')
print(result_df.to_string(index=False))
