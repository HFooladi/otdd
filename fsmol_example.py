from otdd.pytorch.datasets import load_fsmol_data
from otdd.pytorch.distance import DatasetDistance
import pickle
import os
from tqdm import tqdm
import torch


featurizer = 'ChemBERTa-77M-MLM'
with open(f'data/fsmol_{featurizer}_train.pkl', 'rb') as f:
    fsmol_data_train = pickle.load(f)

with open(f'data/fsmol_{featurizer}_test.pkl', 'rb') as f:
    fsmol_data_test = pickle.load(f)

train_chembl_ids = list(fsmol_data_train.keys())
test_chembl_ids = list(fsmol_data_test.keys())

# Load data

distance_matrices = []
for i, src_id in tqdm(enumerate(train_chembl_ids)):
    distance_matirx = []
    loaders_src  = load_fsmol_data(src_id, task='train', featurizer=featurizer)
    for tgt_id in test_chembl_ids:
        loaders_tgt  = load_fsmol_data(tgt_id, task='test', featurizer=featurizer)
        # Instantiate distance
        dist = DatasetDistance(loaders_src, loaders_tgt,
                                inner_ot_method = 'exact',
                                debiased_loss = True,
                                p = 2, entreg = 1e-1,
                                device='cuda:0')

        try:
            d = dist.distance(maxsamples = 1000)
            distance_matirx.append(d.item())
            print(f'iteration_{i}------------------OTDD(ChEMBL1,ChEMBL2)={d:8.2f}')
        except:
            distance_matirx.append(float('NaN'))
            print(f'OTDD(ChEMBL1,ChEMBL2)=NaN')
    
    distance_matrices.append(torch.tensor(distance_matirx))

results = {'train_chembl_ids': train_chembl_ids, 'test_chembl_ids': test_chembl_ids, 'distance_matrices': distance_matrices}

output_path = os.path.join('out', f'fsmol_distance_matrices_{featurizer}.pkl')

with open(output_path, 'wb') as f:
    pickle.dump(results, f)
