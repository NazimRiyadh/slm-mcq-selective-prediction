"""Audit the archived lower-dimensional feature schemas and strict subset.

The script reads the selected primary results supplied with the source package.
It does not rerun model inference or replace archived experiment outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=SCRIPT_DIR/'input_data'/'locked_primary_selected.csv')
parser.add_argument('--out-dir', type=Path, default=SCRIPT_DIR/'analysis_outputs')
parser.add_argument('--bootstrap-reps', type=int, default=50000)
parser.add_argument('--seed', type=int, default=20260623)
args = parser.parse_args()

IN=args.input.resolve(); OUT=args.out_dir.resolve(); OUT.mkdir(parents=True,exist_ok=True)
if not IN.exists(): raise FileNotFoundError(IN)

df=pd.read_csv(IN)
models=['falcon3_7b_instruct','gemma2_9b_it','llama31_8b_instruct','mistral7b_v03','olmo2_7b_instruct','qwen25_7b']
datasets=['arc_challenge','commonsenseqa','hellaswag','mmlu']
df=df[df.model.isin(models)&df.dataset.isin(datasets)].copy()
df['contains_internal_scalar']=df['cheap_cols'].fillna('').str.contains(r'layer|hidden|activation|embedding',case=False,regex=True)
schema=df[['model','dataset','cheap_cols','contains_internal_scalar']].drop_duplicates().sort_values(['model','dataset'])
schema['baseline_schema']=np.where(schema.contains_internal_scalar,'output scores + scalar layer maxima','strict output-only')
schema.to_csv(OUT/'feature_family_schema_audit.csv',index=False)
summary=schema.groupby('baseline_schema').size().rename('n_conditions').reset_index(); summary['percent']=100*summary.n_conditions/schema.shape[0]
summary.to_csv(OUT/'feature_schema_counts.csv',index=False)

metrics=['test_auroc_correct','test_auprc_failure','test_risk_at_80','test_aurc','test_failure_recall_at_20_abstain']
strict=df[~df.contains_internal_scalar].copy()
fam=strict.groupby('family')[metrics].mean().reset_index(); fam.to_csv(OUT/'strict_output_only_family_summary.csv',index=False)
cond_family=strict.groupby(['model','dataset','family'])[metrics].mean().reset_index()
piv=cond_family.pivot(index=['model','dataset'],columns='family',values=metrics)
rows=[]
for model,dataset in piv.index:
    r={'model':model,'dataset':dataset}
    for met in metrics:
        for family in ['confidence_option','cheap_plus_hidden_pca','hidden_pca']:
            r[f'{family}__{met}']=piv.loc[(model,dataset),(met,family)]
        r[f'delta_combined_minus_output__{met}']=piv.loc[(model,dataset),(met,'cheap_plus_hidden_pca')]-piv.loc[(model,dataset),(met,'confidence_option')]
    rows.append(r)
cond=pd.DataFrame(rows); cond.to_csv(OUT/'strict_output_only_condition_results.csv',index=False)

model_labels=sorted(cond.model.unique()); dataset_labels=sorted(cond.dataset.unique())
rng=np.random.default_rng(args.seed)
mi=rng.integers(0,len(model_labels),size=(args.bootstrap_reps,len(model_labels)))
di=rng.integers(0,len(dataset_labels),size=(args.bootstrap_reps,len(dataset_labels)))
results=[]
for met in metrics:
    col=f'delta_combined_minus_output__{met}'
    mat=cond.pivot(index='model',columns='dataset',values=col).loc[model_labels,dataset_labels].to_numpy()
    values=mat[mi[:,:,None],di[:,None,:]].mean(axis=(1,2))
    lodo=[np.delete(mat,j,axis=1).mean() for j in range(mat.shape[1])]
    lomo=[np.delete(mat,i,axis=0).mean() for i in range(mat.shape[0])]
    lo,hi=np.quantile(values,[.025,.975])
    results.append({'metric':met,'mean_delta':mat.mean(),'ci_low':lo,'ci_high':hi,
                    'lodo_min':min(lodo),'lodo_max':max(lodo),'lomo_min':min(lomo),'lomo_max':max(lomo),
                    'n_models':len(model_labels),'n_datasets':len(dataset_labels),'n_conditions':len(cond),
                    'bootstrap_reps':args.bootstrap_reps})
pd.DataFrame(results).to_csv(OUT/'strict_output_only_crossed_deltas.csv',index=False)

weighted=[]
for family,g in strict.groupby('family'):
    row={'family':family}
    for met in metrics: row[met]=np.average(g[met],weights=g.n_test)
    weighted.append(row)
pd.DataFrame(weighted).to_csv(OUT/'strict_output_only_testsize_weighted_summary.csv',index=False)

pivot=strict.pivot_table(index=['model','dataset','seed'],columns='family',values=metrics)
signs=[]
for met in metrics:
    delta=pivot[(met,'cheap_plus_hidden_pca')]-pivot[(met,'confidence_option')]
    signs.append({'metric':met,'n':int(delta.notna().sum()),'positive':int((delta>0).sum()),
                  'zero':int((delta==0).sum()),'negative':int((delta<0).sum()),
                  'mean':delta.mean(),'median':delta.median()})
pd.DataFrame(signs).to_csv(OUT/'strict_output_only_seed_signs_descriptive.csv',index=False)

print(summary.to_string(index=False))
print(f'Strict output-only models: {model_labels}')
print(fam.to_string(index=False))
print(pd.DataFrame(results).to_string(index=False))
