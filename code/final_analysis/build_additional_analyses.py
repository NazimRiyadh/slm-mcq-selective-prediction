from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent

parser = argparse.ArgumentParser(
    description='Reproduce the dependence, selection-sensitivity, and cost-utility analyses added during the MLWA revision.'
)
parser.add_argument(
    '--data-dir', type=Path, default=SCRIPT_DIR / 'input_data',
    help='Directory containing locked_primary_selected.csv, locked_all_candidate_details.csv, and locked_deepseek_stress_selected.csv.'
)
parser.add_argument(
    '--out-dir', type=Path, default=SCRIPT_DIR / 'analysis_outputs',
    help='Directory for derived CSV outputs.'
)
parser.add_argument(
    '--tables-dir', type=Path, default=SOURCE_ROOT / 'tables',
    help='Directory for generated LaTeX tables.'
)
parser.add_argument(
    '--figures-dir', type=Path, default=SOURCE_ROOT / 'figures',
    help='Directory for generated figures.'
)
args = parser.parse_args()

DATA = args.data_dir.resolve()
OUT = args.out_dir.resolve()
TABLES = args.tables_dir.resolve()
FIGS = args.figures_dir.resolve()

required = [
    DATA / 'locked_primary_selected.csv',
    DATA / 'locked_all_candidate_details.csv',
    DATA / 'locked_deepseek_stress_selected.csv',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError('Missing required analysis input(s): ' + ', '.join(missing))

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

MAIN_DS = ['arc_challenge','commonsenseqa','hellaswag','mmlu']
DS_LABEL = {'arc_challenge':'ARC-Challenge','commonsenseqa':'CommonsenseQA','hellaswag':'HellaSwag','mmlu':'MMLU'}
MODEL_LABEL = {
    'falcon3_7b_instruct':'Falcon3-7B',
    'gemma2_9b_it':'Gemma-2-9B-IT',
    'llama31_8b_instruct':'Llama-3.1-8B',
    'mistral7b_v03':'Mistral-7B-v0.3',
    'olmo2_7b_instruct':'OLMo2-7B',
    'qwen25_7b':'Qwen2.5-7B',
}
FAMS = ['confidence_option','cheap_plus_hidden_pca','hidden_pca']
METRICS = ['test_auroc_correct','test_auprc_failure','test_risk_at_80','test_aurc','test_eaurc','test_failure_recall_at_20_abstain']
DELTA_METRICS = ['test_auroc_correct','test_auprc_failure','test_risk_at_80','test_aurc']

sel = pd.read_csv(DATA/'locked_primary_selected.csv')
sel = sel[sel['model'].isin(MODEL_LABEL) & sel['dataset'].isin(MAIN_DS) & sel['family'].isin(FAMS)].copy()
details = pd.read_csv(DATA/'locked_all_candidate_details.csv')
details = details[details['model'].isin(MODEL_LABEL) & details['dataset'].isin(MAIN_DS) & details['family'].isin(FAMS)].copy()

# Condition means after averaging repeated splits.
cond = sel.groupby(['model','dataset','family'], as_index=False)[METRICS + ['accuracy','n','n_test','test_failures']].mean()
deltas = None
for metric in METRICS:
    p = cond.pivot(index=['model','dataset'], columns='family', values=metric)
    d = (p['cheap_plus_hidden_pca'] - p['confidence_option']).rename(metric).reset_index()
    deltas = d if deltas is None else deltas.merge(d, on=['model','dataset'])
deltas.to_csv(OUT/'condition_mean_deltas.csv', index=False)

# Two-way crossed model x dataset bootstrap from seed-averaged condition effects.
def twoway_boot(metric, B=50000, seed=20260623):
    models = sorted(MODEL_LABEL)
    mat = deltas.pivot(index='model', columns='dataset', values=metric).loc[models, MAIN_DS].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        mi = rng.integers(0, len(models), len(models))
        di = rng.integers(0, len(MAIN_DS), len(MAIN_DS))
        vals[b] = mat[np.ix_(mi, di)].mean()
    return float(mat.mean()), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))

cross_rows = []
metric_labels = {
    'test_auroc_correct':'AUROC', 'test_auprc_failure':'AUPRC-fail',
    'test_risk_at_80':'Risk@80%', 'test_aurc':'AURC'
}
for metric in DELTA_METRICS:
    mean, lo, hi = twoway_boot(metric)
    lodo = {DS_LABEL[d]: float(deltas[deltas.dataset != d][metric].mean()) for d in MAIN_DS}
    lomo = {MODEL_LABEL[m]: float(deltas[deltas.model != m][metric].mean()) for m in sorted(MODEL_LABEL)}
    cross_rows.append({
        'metric': metric_labels[metric], 'mean_delta': mean, 'two_way_ci_low': lo, 'two_way_ci_high': hi,
        'lodo_min': min(lodo.values()), 'lodo_max': max(lodo.values()),
        'lomo_min': min(lomo.values()), 'lomo_max': max(lomo.values()),
    })
    pd.DataFrame({'omitted_dataset':list(lodo.keys()), 'mean_delta':list(lodo.values())}).to_csv(OUT/f'lodo_{metric}.csv', index=False)
    pd.DataFrame({'omitted_model':list(lomo.keys()), 'mean_delta':list(lomo.values())}).to_csv(OUT/f'lomo_{metric}.csv', index=False)
pd.DataFrame(cross_rows).to_csv(OUT/'crossed_dependence_summary.csv', index=False)

# Model and dataset mean effects.
model_effect = deltas.groupby('model', as_index=False)[DELTA_METRICS].mean()
model_effect['model_label'] = model_effect.model.map(MODEL_LABEL)
model_effect.to_csv(OUT/'model_mean_deltas.csv', index=False)
dataset_effect = deltas.groupby('dataset', as_index=False)[DELTA_METRICS].mean()
dataset_effect['dataset_label'] = dataset_effect.dataset.map(DS_LABEL)
dataset_effect.to_csv(OUT/'dataset_mean_deltas.csv', index=False)

# Base accuracy and sparse-failure summary. One row per model-dataset-seed.
base = sel[['model','dataset','seed','accuracy','n','n_test','test_failures']].drop_duplicates()
ds_sum = base.groupby('dataset').agg(
    n_examples=('n','first'), base_accuracy_mean=('accuracy','mean'),
    base_accuracy_min=('accuracy','min'), base_accuracy_max=('accuracy','max'),
    test_failures_mean=('test_failures','mean'), test_failures_min=('test_failures','min'),
    test_failures_max=('test_failures','max'), n_condition_seed=('test_failures','size'),
    n_lt10=('test_failures', lambda x: int((x < 10).sum())),
    n_lt20=('test_failures', lambda x: int((x < 20).sum())),
).reset_index()
ds_sum['dataset_label'] = ds_sum.dataset.map(DS_LABEL)
ds_sum.to_csv(OUT/'dataset_accuracy_failure_summary.csv', index=False)

# Equal-condition macro and test-size-weighted averages.
agg_rows = []
for fam in FAMS:
    g = sel[sel.family == fam]
    row = {'family': fam}
    for metric in DELTA_METRICS:
        row[f'{metric}_macro'] = float(g[metric].mean())
        row[f'{metric}_weighted'] = float(np.average(g[metric], weights=g['n_test']))
    # Exact retained-count weighted Risk@80% using ceil(0.8*n_test).
    k = np.ceil(.8 * g['n_test']).astype(int)
    row['risk80_retained_weighted'] = float(np.average(g['test_risk_at_80'], weights=k))
    row['total_retained_across_seed_conditions'] = int(k.sum())
    agg_rows.append(row)
agg = pd.DataFrame(agg_rows)
agg.to_csv(OUT/'macro_testsize_weighted_summary.csv', index=False)

# Selection sensitivity: original full grid selected by val AUROC; full grid selected by val AURC/Risk80;
# equal-budget fixed PCA=32 with 3 C candidates, matching cheap family's 3 candidates.
def select_rows(frame, criterion, maximize=True, fixed_pca=None):
    f = frame.copy()
    if fixed_pca is not None:
        keep = (f.family == 'confidence_option') | (f.pca_dim == fixed_pca)
        f = f[keep]
    sort_asc = not maximize
    f = f.sort_values(['model','dataset','seed','family',criterion], ascending=[True,True,True,True,sort_asc])
    return f.groupby(['model','dataset','seed','family'], as_index=False).first()

schemes = {
    'Validation AUROC, full grid': ('val_auroc_correct', True, None),
    'Validation AURC, full grid': ('val_aurc', False, None),
    'Validation Risk@80%, full grid': ('val_risk_at_80', False, None),
    'Validation AUROC, equal budget (PCA=32)': ('val_auroc_correct', True, 32),
}
ss_rows = []
for name, (crit, maxit, fpca) in schemes.items():
    s = select_rows(details, crit, maxit, fpca)
    for metric in DELTA_METRICS:
        p = s[s.family.isin(['confidence_option','cheap_plus_hidden_pca'])].pivot_table(
            index=['model','dataset','seed'], columns='family', values=metric, aggfunc='first').dropna()
        d = p['cheap_plus_hidden_pca'] - p['confidence_option']
        ss_rows.append({'selection_scheme':name, 'metric':metric_labels[metric], 'n_pairs':len(d),
                        'mean_delta':float(d.mean()), 'median_delta':float(d.median()),
                        'positive':int((d>0).sum()), 'negative':int((d<0).sum())})
selection_sensitivity = pd.DataFrame(ss_rows)
selection_sensitivity.to_csv(OUT/'selection_criterion_and_budget_sensitivity.csv', index=False)

# Validation-to-test optimism/degradation among selected configurations.
gap_rows=[]
for fam in FAMS:
    g=sel[sel.family==fam]
    gap = g['test_auroc_correct'] - g['val_auroc_correct']
    gap_rows.append({'family':fam,'mean_test_minus_val_auroc':float(gap.mean()),
                     'sd':float(gap.std(ddof=1)), 'median':float(gap.median())})
pd.DataFrame(gap_rows).to_csv(OUT/'validation_to_test_auroc_gap.csv', index=False)

# Hyperparameter selection frequencies.
freqlist=[]
for fam in ['cheap_plus_hidden_pca','hidden_pca','confidence_option']:
    g=sel[sel.family==fam]
    fr=g.groupby(['candidate','C','pca_dim']).size().reset_index(name='selected_count')
    fr['family']=fam
    freqlist.append(fr)
pd.concat(freqlist, ignore_index=True).to_csv(OUT/'selected_hyperparameter_frequencies.csv', index=False)

# Practical-effect sensitivity; post hoc thresholds, explicitly not equivalence tests.
# Beneficial directions: + for AUROC/AUPRC; - for risk/AURC.
thresholds = {
    'test_auroc_correct': 0.01,
    'test_auprc_failure': 0.01,
    'test_risk_at_80': -0.01,
    'test_aurc': -0.005,
}
practical=[]
for metric, threshold in thresholds.items():
    vals=deltas[metric]
    if metric in ['test_auroc_correct','test_auprc_failure']:
        meaningful = vals >= threshold
        modest_or_better = vals > 0
    else:
        meaningful = vals <= threshold
        modest_or_better = vals < 0
    practical.append({'metric':metric_labels[metric], 'posthoc_beneficial_threshold':threshold,
                      'mean_delta':float(vals.mean()), 'conditions_any_benefit':int(modest_or_better.sum()),
                      'conditions_meeting_threshold':int(meaningful.sum()), 'n_conditions':len(vals)})
pd.DataFrame(practical).to_csv(OUT/'practical_effect_sensitivity.csv', index=False)

# DeepSeek boundary and pooled-without/with summaries.
deep = pd.read_csv(DATA/'locked_deepseek_stress_selected.csv')
deep_base = deep[['dataset','accuracy','n','n_failures']].drop_duplicates().copy()
deep_base['chance_level'] = deep_base.dataset.map({'arc_challenge':.25,'commonsenseqa':.20,'hellaswag':.25,'mmlu':.25})
deep_base['margin_above_chance'] = deep_base.accuracy-deep_base.chance_level
deep_base.to_csv(OUT/'deepseek_base_accuracy_vs_chance.csv', index=False)
# paired combined-cheap averages for stress test
p=deep[deep.family.isin(['confidence_option','cheap_plus_hidden_pca'])].pivot_table(index=['dataset','seed'],columns='family',values=DELTA_METRICS,aggfunc='first')
deep_delta=[]
for metric in DELTA_METRICS:
    d=(p[(metric,'cheap_plus_hidden_pca')]-p[(metric,'confidence_option')]).dropna()
    deep_delta.append({'metric':metric_labels[metric], 'mean_delta':float(d.mean()), 'n_pairs':len(d)})
pd.DataFrame(deep_delta).to_csv(OUT/'deepseek_delta_summary.csv', index=False)

# Cost-utility mapping by model. Latency table values from manuscript audit.
latency = pd.DataFrame([
    ('qwen25_7b',251.3,272.7,5.33,5.35),
    ('gemma2_9b_it',270.4,280.5,2.68,2.70),
    ('falcon3_7b_instruct',259.1,272.5,6.68,6.69),
    ('llama31_8b_instruct',266.3,257.8,2.47,2.49),
    ('mistral7b_v03',238.8,243.2,5.10,5.11),
    ('olmo2_7b_instruct',274.6,279.3,6.57,6.59),
],columns=['model','cheap_ms','hidden_ms','cheap_vram_gb','hidden_vram_gb'])
latency['latency_overhead_pct']=(latency.hidden_ms/latency.cheap_ms-1)*100
risk_by_model=deltas.groupby('model',as_index=False)['test_risk_at_80'].mean().rename(columns={'test_risk_at_80':'risk_delta'})
costutil=latency.merge(risk_by_model,on='model')
costutil['additional_retained_errors_per_1000']=costutil.risk_delta*800
costutil['model_label']=costutil.model.map(MODEL_LABEL)
costutil.to_csv(OUT/'model_level_cost_utility.csv',index=False)

# --- LaTeX tables ---
def fmt_signed(x, nd=4):
    return f'{x:+.{nd}f}'
def ci_str(lo,hi):
    return f'[{lo:+.4f}, {hi:+.4f}]'

# Dataset difficulty table for main text.
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Primary-dataset difficulty and held-out failure prevalence. Accuracy is averaged over the six models; failure counts summarize the 30 model--seed test splits per dataset. Sparse-failure rows are most common for ARC-Challenge and motivate prevalence-aware interpretation of AUPRC.}',
       r'\label{tab:dataset-failure-summary}',r'\begin{adjustbox}{width=\textwidth}',
       r'\begin{tabular}{lrrrrrr}',r'\toprule',
       r'Dataset & $n$ & Accuracy mean [range] & Test failures mean [range] & $<10$ & $<20$ & Splits \\',r'\midrule']
for _,r in ds_sum.iterrows():
    lines.append(f"{r.dataset_label} & {int(r.n_examples)} & {r.base_accuracy_mean:.3f} [{r.base_accuracy_min:.3f}, {r.base_accuracy_max:.3f}] & {r.test_failures_mean:.1f} [{int(r.test_failures_min)}, {int(r.test_failures_max)}] & {int(r.n_lt10)} & {int(r.n_lt20)} & {int(r.n_condition_seed)} \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
(TABLES/'table_dataset_failure_summary.tex').write_text('\n'.join(lines))

# Crossed dependence table.
cr=pd.DataFrame(cross_rows)
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Dependence-aware sensitivity for PCA-hidden augmentation relative to confidence/option features. Effects are computed after averaging the five repeated splits within each model--dataset condition. The two-way bootstrap resamples model and dataset clusters independently (50,000 replicates). Leave-one-dataset-out (LODO) and leave-one-model-out (LOMO) columns show the range of recomputed mean effects. Positive deltas favor augmentation for AUROC/AUPRC and disfavor it for Risk@80\%/AURC.}',
       r'\label{tab:crossed-dependence}',r'\begin{adjustbox}{width=\textwidth}',
       r'\begin{tabular}{lrrrr}',r'\toprule',r'Metric & Mean $\Delta$ & Two-way 95\% CI & LODO range & LOMO range \\',r'\midrule']
for _,r in cr.iterrows():
    lines.append(f"{r.metric} & {fmt_signed(r.mean_delta)} & {ci_str(r.two_way_ci_low,r.two_way_ci_high)} & [{fmt_signed(r.lodo_min)}, {fmt_signed(r.lodo_max)}] & [{fmt_signed(r.lomo_min)}, {fmt_signed(r.lomo_max)}] \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
(TABLES/'table_crossed_dependence.tex').write_text('\n'.join(lines))

# Macro vs weighted.
family_label={'confidence_option':'Confidence/option','cheap_plus_hidden_pca':'Confidence/option + PCA hidden','hidden_pca':'PCA hidden only'}
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Equal-condition macro averages and test-size-weighted averages on the 24-condition MCQ suite. Test-size weighting reduces the influence of the smaller ARC-Challenge test splits; it is not a pooled AUROC/AUPRC because repeated test splits overlap.}',
       r'\label{tab:macro-weighted}',r'\begin{adjustbox}{width=\textwidth}',
       r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
       r'& \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{AUPRC-fail} & \multicolumn{2}{c}{Risk@80\%} & \multicolumn{2}{c}{AURC} \\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
       r'Selector & Macro & Weighted & Macro & Weighted & Macro & Weighted & Macro & Weighted \\',r'\midrule']
for _,r in agg.iterrows():
    lines.append(f"{family_label[r.family]} & {r.test_auroc_correct_macro:.3f} & {r.test_auroc_correct_weighted:.3f} & {r.test_auprc_failure_macro:.3f} & {r.test_auprc_failure_weighted:.3f} & {r.test_risk_at_80_macro:.3f} & {r.risk80_retained_weighted:.3f} & {r.test_aurc_macro:.3f} & {r.test_aurc_weighted:.3f} \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
(TABLES/'table_macro_weighted.tex').write_text('\n'.join(lines))

# Selection sensitivity table.
ss=selection_sensitivity
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Selection-objective and search-budget sensitivity for combined PCA-hidden augmentation minus the confidence/option selector. Equal-budget analysis fixes PCA dimension at 32, leaving three regularization candidates in each family. Positive deltas favor augmentation for AUROC/AUPRC and disfavor it for Risk@80\%/AURC.}',
       r'\label{tab:selection-sensitivity}',r'\begin{adjustbox}{width=\textwidth}',
       r'\begin{tabular}{lrrrr}',r'\toprule',r'Selection scheme & $\Delta$ AUROC & $\Delta$ AUPRC & $\Delta$ Risk@80\% & $\Delta$ AURC \\',r'\midrule']
for scheme in schemes:
    vals={row.metric:row.mean_delta for row in ss[ss.selection_scheme==scheme].itertuples()}
    label=scheme.replace('%',r'\%')
    lines.append(f"{label} & {fmt_signed(vals['AUROC'])} & {fmt_signed(vals['AUPRC-fail'])} & {fmt_signed(vals['Risk@80%'])} & {fmt_signed(vals['AURC'])} \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
(TABLES/'table_selection_sensitivity.tex').write_text('\n'.join(lines))

# Practical effect sensitivity table.
prac=pd.DataFrame(practical)
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Post hoc practical-effect sensitivity across 24 seed-averaged model--dataset conditions. Thresholds are interpretive, not preregistered equivalence margins: +0.01 AUROC/AUPRC, -0.01 Risk@80\% (eight fewer retained errors per 1000 queries at 80\% coverage), and -0.005 AURC.}',
       r'\label{tab:practical-effect}',r'\begin{tabular}{lrrrr}',r'\toprule',
       r'Metric & Mean $\Delta$ & Beneficial threshold & Any benefit & Meets threshold \\',r'\midrule']
for _,r in prac.iterrows():
    lines.append(f"{r.metric} & {fmt_signed(r.mean_delta)} & {fmt_signed(r.posthoc_beneficial_threshold,3)} & {int(r.conditions_any_benefit)}/24 & {int(r.conditions_meeting_threshold)}/24 \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(TABLES/'table_practical_effect_sensitivity.tex').write_text('\n'.join(lines))

# DeepSeek table.
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{DeepSeek-R1-Distill-Qwen-7B forced-choice compatibility audit. Accuracy is only slightly above the task-specific chance level under direct answer-letter likelihood scoring. The designation as a protocol stress test was made after this interface audit and is therefore reported transparently as post hoc; all reliability results are retained in the supplement.}',
       r'\label{tab:deepseek-compatibility}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Dataset & $n$ & Accuracy & Chance & Margin above chance \\',r'\midrule']
for _,r in deep_base.iterrows():
    lines.append(f"{DS_LABEL[r.dataset]} & {int(r.n)} & {r.accuracy:.3f} & {r.chance_level:.2f} & {r.margin_above_chance:+.3f} \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(TABLES/'table_deepseek_compatibility.tex').write_text('\n'.join(lines))

# Validation gap table.
gap=pd.read_csv(OUT/'validation_to_test_auroc_gap.csv')
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Validation-to-test AUROC change for the validation-selected configuration. More negative values indicate greater validation optimism. The richer feature families show larger average degradation, consistent with the need for locked selection and equal-budget sensitivity.}',
       r'\label{tab:validation-gap}',r'\begin{tabular}{lrrr}',r'\toprule',r'Family & Mean test$-$validation & SD & Median \\',r'\midrule']
for _,r in gap.iterrows():
    lines.append(f"{family_label[r.family]} & {r.mean_test_minus_val_auroc:+.4f} & {r.sd:.4f} & {r['median']:+.4f} \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(TABLES/'table_validation_gap.tex').write_text('\n'.join(lines))

# Model/dataset effects full table for supplement.
lines=[r'\begin{table}[t]',r'\centering',r'\small',
       r'\caption{Seed-averaged combined-minus-cheap effects by model and dataset. Positive values favor augmentation for AUROC/AUPRC and disfavor it for Risk@80\%/AURC.}',
       r'\label{tab:model-dataset-effects}',r'\begin{adjustbox}{width=\textwidth}',
       r'\begin{tabular}{llrrrr}',r'\toprule',r'Model & Dataset & $\Delta$ AUROC & $\Delta$ AUPRC & $\Delta$ Risk@80\% & $\Delta$ AURC \\',r'\midrule']
for _,r in deltas.sort_values(['model','dataset']).iterrows():
    lines.append(f"{MODEL_LABEL[r.model]} & {DS_LABEL[r.dataset]} & {r.test_auroc_correct:+.4f} & {r.test_auprc_failure:+.4f} & {r.test_risk_at_80:+.4f} & {r.test_aurc:+.4f} \\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
(TABLES/'table_model_dataset_effects.tex').write_text('\n'.join(lines))

# --- Figures ---
# Condition heatmap: AUROC delta.
heat = deltas.pivot(index='model', columns='dataset', values='test_auroc_correct').loc[sorted(MODEL_LABEL), MAIN_DS]
fig, ax = plt.subplots(figsize=(7.4, 4.8))
im = ax.imshow(heat.values, aspect='auto')
ax.set_xticks(range(len(MAIN_DS)), [DS_LABEL[d] for d in MAIN_DS], rotation=25, ha='right')
ax.set_yticks(range(len(heat.index)), [MODEL_LABEL[m] for m in heat.index])
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f'{heat.iloc[i,j]:+.3f}', ha='center', va='center', fontsize=8)
ax.set_title('Seed-averaged AUROC change: PCA-hidden augmentation minus cheap selector')
cb=fig.colorbar(im, ax=ax)
cb.set_label('AUROC change')
fig.tight_layout()
fig.savefig(FIGS/'fig_condition_auroc_delta_heatmap.pdf', bbox_inches='tight')
fig.savefig(FIGS/'fig_condition_auroc_delta_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Model-level cost utility scatter.
fig, ax = plt.subplots(figsize=(7.2,4.8))
ax.scatter(costutil.latency_overhead_pct, costutil.additional_retained_errors_per_1000, s=55)
for r in costutil.itertuples():
    ax.annotate(r.model_label, (r.latency_overhead_pct, r.additional_retained_errors_per_1000),
                xytext=(4,4), textcoords='offset points', fontsize=8)
ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)
ax.set_xlabel('Measured hidden-readout latency overhead (%)')
ax.set_ylabel('Additional retained errors per 1000 queries at 80% coverage')
ax.set_title('Model-level cost-utility diagnostic')
fig.tight_layout()
fig.savefig(FIGS/'fig_cost_utility_pareto.pdf', bbox_inches='tight')
fig.savefig(FIGS/'fig_cost_utility_pareto.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print('Wrote analyses to', OUT)
print(pd.DataFrame(cross_rows).to_string(index=False))
print('\nSelection sensitivity\n', selection_sensitivity.to_string(index=False))
print('\nMacro/weighted\n', agg.to_string(index=False))
print('\nPractical\n', pd.DataFrame(practical).to_string(index=False))
