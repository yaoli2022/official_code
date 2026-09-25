import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv

BASE_DIR         = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNHEALTHY_DIR    = os.path.join(BASE_DIR, 'data/unhelathy-dataset/Unhealthy Normalized')
ALTUMAGE_CPGS    = os.path.join(BASE_DIR, 'data/multi_platform_cpgs.pkl')
CPG_INFO_PATH    = os.path.join(BASE_DIR, 'data/cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
DATA_DIR         = os.path.join(BASE_DIR, 'data/all-organs4/all_organs')
BASELINES_DIR    = os.path.join(BASE_DIR, 'baselines')
CKPT_OURS        = os.path.join(BASE_DIR, 'checkpoints/pna_gnn_stat_fold2.pth')
CKPT_BASELINE    = os.path.join(BASE_DIR, 'checkpoints/pna_gnn_fold2.pth')
HORVATH_COEF     = os.path.join(BASELINES_DIR, 'coefficients.csv')
ALTUMAGE_H5      = os.path.join(BASELINES_DIR, 'AltumAge.h5')
ALTUMAGE_SCALER  = os.path.join(BASELINES_DIR, 'scaler.pkl')
OUTPUT_DIR       = BASELINES_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

THRESHOLD_CORR      = 0.70
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD  = THRESHOLD_CORR - 0.04
THRESHOLD_DIST      = 1e5
EDGE_DIM            = 3
K_FOLDS             = 5
DESIRED_FOLD        = 2
ADULT_AGE           = 20

DISEASE_MAP = {
    'GSE19711':      'Ovarian Cancer',
    'GSE41037':      'Schizophrenia',
    'GSE27044':      'Osteoporosis',
    'GSE99624':      'Osteoporosis',
    'E-GEOD-44763':  'Osteoporosis',
    'GSE77241':      'Osteoporosis',
}

print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
print(f'  {len(AltumAge_cpgs)} CpG sites')

print('Loading CpG site information...')
information = pd.read_csv(CPG_INFO_PATH, skiprows=7, low_memory=False)
information.dropna(subset=['Chr'], inplace=True)
information[['start', 'end']] = (
    information.CPG_ISLAND_LOCATIONS
    .fillna('0:0-0').str.split(':').str[1]
    .str.split('-', expand=True).astype(int)
)
information['CPG_ISLAND']     = information['CPG_ISLAND'].astype(int)
information['CPG_ISLAND_LEN'] = information.end - information.start
information.MapInfo            = information.MapInfo.astype(int)

scaler_mm = MinMaxScaler()
for col in ['MapInfo', 'TSS_Coordinate']:
    information[f'Normalized_{col}'] = information.groupby('Chr')[col].transform(
        lambda x: scaler_mm.fit_transform(x.values.reshape(-1,1)).flatten()
    )
information = pd.get_dummies(information, columns=['Gene_Strand'])
for col in ['start', 'end', 'CPG_ISLAND_LEN']:
    information[col] = information.groupby('Chr')[col].transform(
        lambda x: scaler_mm.fit_transform(x.values.reshape(-1,1)).flatten()
    )
information = pd.get_dummies(information, columns=['Next_Base'])
information['Distance_to_TSS'] = information.groupby('Chr')['Distance_to_TSS'].transform(
    lambda x: scaler_mm.fit_transform(x.values.reshape(-1,1)).flatten()
)
information['Distance_to_TSS'] = information['Distance_to_TSS'].fillna(1)
information = pd.get_dummies(information, columns=['SourceStrand'])
Chrom = information.Chr.tolist()
information = pd.get_dummies(information, columns=['Chr'])
information['Chr'] = Chrom
information.index  = information.IlmnID

NODE_FEATURE_DICT = {
    1: 'CPG_ISLAND', 2: 'CPG_ISLAND_LEN', 3: 'Distance_to_TSS',
    4: ['Next_Base_A', 'Next_Base_C', 'Next_Base_T'],
    5: 'start', 6: 'end', 7: 'Normalized_TSS_Coordinate',
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1,23)]
}
selected_features        = '1,2,3,4,5,6,8'
selected_feature_numbers = list(map(int, selected_features.split(',')))
user_selected_features   = []
for key in selected_feature_numbers:
    feat = NODE_FEATURE_DICT[key]
    if isinstance(feat, list):
        user_selected_features.extend(feat)
    else:
        user_selected_features.append(feat)

INPUT_DIM = len(user_selected_features) + 1
filtered_information = information[information.IlmnID.isin(AltumAge_cpgs)]

node2cpg = {}
cpg2node = {}
for i, c in enumerate(AltumAge_cpgs):
    cpg2node[c] = i
    node2cpg[i] = c

print('\nLoading healthy training data for graph construction...')

def select_healthy(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

from sklearn.model_selection import train_test_split, KFold

train_frames = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pkl'):
        df = select_healthy(pd.read_pickle(os.path.join(DATA_DIR, filename)))
        if len(df) <= 0:
            continue
        tr, _ = train_test_split(df, test_size=0.2, random_state=42)
        train_frames.append(tr)

train_combined = pd.concat(train_frames)
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx].sample(frac=1, random_state=42)
    break

X_fold_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
print(f'  Training samples for graph: {len(X_fold_train)}')

print('\nBuilding co-methylation graph...')
chromosomes_arr = filtered_information.Chr.values
genes           = filtered_information.Symbol.values
chromosome_adj  = (chromosomes_arr[:, None] == chromosomes_arr).astype(np.float32)
base_pair       = filtered_information.MapInfo.values
distance_matrix = squareform(pdist(base_pair.reshape(-1,1)))
genes_arr       = np.array(genes)
genes_adj       = (genes_arr[:, None] == genes_arr).astype(np.float32)
adj             = np.corrcoef(X_fold_train.to_numpy(), rowvar=False)

src, dst = np.where(
    (((chromosome_adj == 1) &
      ((np.abs(adj) > SECONDARY_THRESHOLD) |
       ((np.abs(distance_matrix) < THRESHOLD_DIST) & (np.abs(adj) > TERTIARY_THRESHOLD))))
     | (np.abs(adj) > THRESHOLD_CORR))
    & (np.arange(adj.shape[0])[:, None] != np.arange(adj.shape[1]))
)
weights    = np.column_stack([adj[src,dst], chromosome_adj[src,dst], genes_adj[src,dst]])
edge_index = torch.stack([torch.tensor(src, dtype=torch.int64),
                           torch.tensor(dst, dtype=torch.int64)], dim=0)
edge_attr  = torch.tensor(weights, dtype=torch.float32).reshape(-1, EDGE_DIM)
print(f'  Edges: {edge_index.shape[1]}')

class DNASequenceProcessor:
    def extract_statistical_features(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros(8, dtype=np.float32)
        seq     = seq.replace('[CG]', 'CG').upper()
        n       = len(seq)
        gc      = (seq.count('G') + seq.count('C')) / max(n, 1)
        cpg_dens= seq.count('CG') / max(n-1, 1)
        up      = seq[:60]
        gc_up   = (up.count('G') + up.count('C')) / max(len(up), 1)
        dn      = seq[62:]
        gc_dn   = (dn.count('G') + dn.count('C')) / max(len(dn), 1)
        ctx     = seq[55:65] if len(seq) >= 65 else seq
        return np.array([gc, cpg_dens, gc_up, gc_dn,
                         ctx.count('A')/max(len(ctx),1),
                         ctx.count('T')/max(len(ctx),1),
                         ctx.count('C')/max(len(ctx),1),
                         ctx.count('G')/max(len(ctx),1)], dtype=np.float32)

    def extract_all_statistical_features(self, sequences):
        return np.array([self.extract_statistical_features(s) for s in sequences],
                        dtype=np.float32)

processor = DNASequenceProcessor()
seq_data  = processor.extract_all_statistical_features(
    filtered_information['TopGenomicSeq'].values)
print(f'seq_data shape: {seq_data.shape}')

print('\nLoading unhealthy dataset...')

def select_unhealthy(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    a['age'] = pd.to_numeric(a['age'], errors='coerce')
    return a.dropna(subset=['age'])

unhealthy_frames = []
for filename in os.listdir(UNHEALTHY_DIR):
    if filename.endswith('.pkl'):
        df = select_unhealthy(pd.read_pickle(os.path.join(UNHEALTHY_DIR, filename)))
        if len(df) > 0:
            unhealthy_frames.append(df)

unhealthy_data = pd.concat(unhealthy_frames)
unhealthy_data['disease'] = unhealthy_data['dataset'].map(DISEASE_MAP).fillna('Unknown')

X_unhealthy = unhealthy_data.drop(
    columns=['age','gender','dataset','tissue_type','disease']).astype('float')
y_unhealthy = unhealthy_data['age'].astype('float')
print(f'  Unhealthy samples: {len(unhealthy_data)}')
print(unhealthy_data['disease'].value_counts())

def make_loader(X_, y_, batch_size=1):
    seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
    graphs = []
    for row in range(len(X_)):
        x = pd.concat([X_.iloc[row, :],
                        filtered_information[user_selected_features]],
                       axis=1, join='inner')
        x_t = torch.tensor(x.to_numpy().astype('float'), dtype=torch.float32)
        y_t = torch.tensor(float(y_.iloc[row]), dtype=torch.float32)
        graphs.append(Data(x=x_t, x_seq=seq_tensor, y=y_t,
                           edge_index=edge_index, edge_attr=edge_attr))
    return DataLoader(graphs, batch_size=batch_size)

class BaselineNet(nn.Module):
    def __init__(self, deg, original_dim=10, num_cpgs=20318):
        super().__init__()
        aggregators = ['mean','max','std','min']
        scalers     = ['identity','amplification','attenuation']
        self.LastLayer = PNAConv(
            in_channels=original_dim, out_channels=1,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=3, towers=1, pre_layers=1, post_layers=1,
            divide_input=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs,1024), nn.ReLU(),
            nn.Linear(1024,656),      nn.SELU(),
            nn.Linear(656,256),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,124),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(124,64),        nn.SELU(),
            nn.Linear(64,32),         nn.ReLU(),
            nn.Linear(32,8),          nn.ReLU(),
            nn.Linear(8,1)
        )

    def forward(self, x_original, x_seq, edge_index, edge_attr, batch):
        x = self.LastLayer(x_original, edge_index, edge_attr)
        x = F.relu(x)
        return self.mlp(x.T).flatten()

class HierarchicalNet(nn.Module):
    def __init__(self, deg, original_dim=10, num_cpgs=20318):
        super().__init__()
        self.importance_gate = nn.Sequential(
            nn.Linear(8,16), nn.ReLU(), nn.Linear(16,1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(8,8), nn.ReLU(), nn.Linear(8,2)
        )
        aggregators = ['mean','max','std','min']
        scalers     = ['identity','amplification','attenuation']
        self.LastLayer = PNAConv(
            in_channels=original_dim+2, out_channels=1,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=3, towers=1, pre_layers=1, post_layers=1,
            divide_input=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs,1024), nn.ReLU(),
            nn.Linear(1024,656),      nn.SELU(),
            nn.Linear(656,256),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,124),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(124,64),        nn.SELU(),
            nn.Linear(64,32),         nn.ReLU(),
            nn.Linear(32,8),          nn.ReLU(),
            nn.Linear(8,1)
        )

    def forward(self, x_original, x_seq, edge_index, edge_attr, batch):
        imp  = self.importance_gate(x_seq)
        meth = x_original[:, 0:1] * imp
        proj = self.seq_proj(x_seq)
        x    = torch.cat([meth, x_original[:,1:], proj], dim=-1)
        x    = self.LastLayer(x, edge_index, edge_attr)
        x    = F.relu(x)
        return self.mlp(x.T).flatten()

def compute_deg(loader):
    max_degree = -1
    for data in loader:
        max_degree = max(max_degree, int(data.edge_index[1].max()))
        break
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in loader.dataset:
        d = torch.bincount(data.edge_index[1].cpu(), minlength=max_degree+1)
        deg += d
    return deg

def gnn_predict(ckpt_path, loader, deg_ref, model_class=HierarchicalNet):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        saved_deg  = ckpt.get('deg', deg_ref)
    else:
        state_dict = ckpt
        saved_deg  = deg_ref

    model = model_class(deg=saved_deg.to('cpu'),
                        original_dim=INPUT_DIM,
                        num_cpgs=len(AltumAge_cpgs)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    with torch.no_grad():
        for data in tqdm(loader, desc='  GNN inference'):
            data = data.to(device)
            out  = model(data.x, data.x_seq, data.edge_index, data.edge_attr, data.batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()

def anti_transform_age(x):
    x = np.array(x, dtype=float)
    return np.where(
        x < 0,
        np.exp(x + np.log(ADULT_AGE + 1)) - 1,
        x * (ADULT_AGE + 1) + ADULT_AGE
    )

print('\nBuilding unhealthy DataLoader...')
unhealthy_loader = make_loader(X_unhealthy, y_unhealthy, batch_size=1)
deg_ref = compute_deg(unhealthy_loader)

results = pd.DataFrame({
    'true_age':  y_unhealthy.values,
    'dataset':   unhealthy_data['dataset'].values,
    'disease':   unhealthy_data['disease'].values,
})

print('\n--- Horvath (2013) ---')
coef_data    = pd.read_csv(HORVATH_COEF)
intercept    = coef_data[coef_data.CpGmarker=='(Intercept)']['CoefficientTraining'].values[0]
coef_df      = coef_data[coef_data.CpGmarker!='(Intercept)']
horvath_cpgs = np.array(coef_df['CpGmarker'])
coefs        = np.array(coef_df['CoefficientTraining'])

available = [c for c in horvath_cpgs if c in X_unhealthy.columns]
X_h = pd.DataFrame(0.0, index=X_unhealthy.index, columns=horvath_cpgs)
for cpg in available:
    X_h[cpg] = X_unhealthy[cpg].values

hm = linear_model.LinearRegression()
hm.coef_      = coefs
hm.intercept_ = intercept
pred_horvath  = np.clip(anti_transform_age(hm.predict(X_h)), 0, 120)
results['Horvath'] = pred_horvath
print(f'  MAE={mean_absolute_error(y_unhealthy, pred_horvath):.3f}')

print('\n--- AltumAge (2022) --- SKIPPED (run inference_unhealthy_altumage.py in a TensorFlow environment)')
results['AltumAge'] = np.nan

print('\n--- PNA-GNN baseline ---')
pred_pna = gnn_predict(CKPT_BASELINE, unhealthy_loader, deg_ref, model_class=BaselineNet)
results['PNA-GNN'] = pred_pna
print(f'  MAE={mean_absolute_error(y_unhealthy, pred_pna):.3f}')

print('\n--- PNA-GNN + Statistical Features (Ours) ---')
pred_ours = gnn_predict(CKPT_OURS, unhealthy_loader, deg_ref)
results['PNA-GNN+Stat'] = pred_ours
print(f'  MAE={mean_absolute_error(y_unhealthy, pred_ours):.3f}')

model_cols = ['Horvath', 'AltumAge', 'PNA-GNN', 'PNA-GNN+Stat']
for col in model_cols:
    results[f'{col}_accel'] = results[col] - results['true_age']

pred_csv = os.path.join(OUTPUT_DIR, 'unhealthy_predictions.csv')
results.to_csv(pred_csv, index=False)
print(f'\nSample-level predictions saved: {pred_csv}')

print('\n' + '='*60)
print('AGE ACCELERATION SUMMARY (mean ± std)')
print('='*60)

accel_cols = [f'{col}_accel' for col in model_cols]
summary = results.groupby('disease')[accel_cols].agg(['mean','std']).round(3)
print(summary.to_string())

summary_flat = []
for disease in results['disease'].unique():
    sub = results[results['disease'] == disease]
    row = {'disease': disease, 'n': len(sub)}
    for col in model_cols:
        row[f'{col}_mean_accel'] = round(float((sub[col] - sub['true_age']).mean()), 3)
        row[f'{col}_std_accel']  = round(float((sub[col] - sub['true_age']).std()), 3)
    summary_flat.append(row)

summary_df = pd.DataFrame(summary_flat)
accel_csv  = os.path.join(OUTPUT_DIR, 'unhealthy_acceleration.csv')
summary_df.to_csv(accel_csv, index=False)
print(f'\nSummary saved: {accel_csv}')
print(summary_df.to_string(index=False))
