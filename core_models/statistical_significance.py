
test_loader = create_data_loaders(params)
import os
import pickle

dataset = test_loader.dataset
volume_ids = []

for mapping in dataset.index_map:
    # mapping is usually a tuple like (group_index, slice_index)
    group_idx = mapping[0] 
    
    # Use the group index to get the actual volume/file name
    vol_id = dataset.groups[group_idx]
    volume_ids.append(str(vol_id))

print(f"Extracted {len(volume_ids)} volume IDs.")
print(f"First 5 IDs for verification: {volume_ids[:5]}")

# Save it to your stats directory
stats_dir = './stat_results'
os.makedirs(stats_dir, exist_ok=True)
with open(os.path.join(stats_dir, 'volume_ids.pkl'), 'wb') as f:
    pickle.dump(volume_ids, f)
    
print("Saved volume_ids.pkl successfully!")


import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

stats_dir = './stat_results'

baselines = {
    'Poisson 1-NEX': 'Poisson_1NEX',
    'Poisson 3-NEX': 'Poisson_3NEX',
    'LOUPE 1-NEX':   'LOUPE_1NEX',
    'LOUPE 2-NEX':   'LOUPE_2NEX',
    'LOUPE 3-NEX':   'LOUPE_3NEX',
}

higher_is_better = {
    'psnr': True, 'ssim': True, 'fsim': True,
}

def load_metrics(method_tag, current_tag_suffix):
    path = os.path.join(stats_dir, f'{method_tag}_{current_tag_suffix}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load the volume IDs mapping we just created
with open(os.path.join(stats_dir, 'volume_ids.pkl'), 'rb') as f:
    volume_ids = pickle.load(f)

contrasts = ['T1w', 'T2w']
r_values = ['166', '2', '3']
version = 'c'

for contrast in contrasts:
    for r_name in r_values:
        tag_suffix = f'{contrast}_x{r_name}{version}'
        print(f"Processing: {tag_suffix}...")

        try:
            nexop_raw = load_metrics('NexOP', tag_suffix)
        except FileNotFoundError:
            continue

        rows = []
        for baseline_name, baseline_tag in baselines.items():
            try:
                baseline_raw = load_metrics(baseline_tag, tag_suffix)
            except FileNotFoundError:
                continue

            for metric, better_high in higher_is_better.items():
                if metric not in nexop_raw or metric not in baseline_raw:
                    continue
                
                # Aggregate per subject
                df = pd.DataFrame({
                    'volume_id': volume_ids,
                    'nexop': nexop_raw[metric],
                    'baseline': baseline_raw[metric]
                })
                
                # Average slices for each unique volume_id
                subject_agg = df.groupby('volume_id').mean()
                
                a = subject_agg['nexop'].values
                b = subject_agg['baseline'].values

                diff = a - b
                try:
                    # Supervisor Request 2 & 1: Two-sided test on subject-level data
                    _, p_wilcoxon = stats.wilcoxon(a, b, alternative='two-sided')
                except ValueError:
                    p_wilcoxon = 1.0  
                
                # Supervisor Request 4: Median effect size (NexOP - baseline)
                median_effect = np.median(diff)

                rows.append({
                    'baseline': baseline_name,
                    'metric': metric.upper(),
                    'median_effect': median_effect,
                    'p_wilcoxon': p_wilcoxon,
                })

        if not rows:
            continue

        results_df = pd.DataFrame(rows)

        # Supervisor Request 3: Holm-Bonferroni correction (across the 5 baselines per metric)
        corrected = []
        for metric in results_df['metric'].unique():
            mask = results_df['metric'] == metric
            reject, p_holm, _, _ = multipletests(results_df.loc[mask, 'p_wilcoxon'], method='holm')
            corrected.append(pd.DataFrame(
                {'p_wilcoxon_holm': p_holm, 'significant (α=0.05)': reject},
                index=results_df[mask].index,
            ))
        
        results_df = results_df.join(pd.concat(corrected))
        save_path = f'stats_results_{tag_suffix}.csv'
        results_df.to_csv(save_path, index=False)
        print(f"  Saved -> {save_path}")


import pandas as pd
import glob

csv_files = glob.glob('stats_results_*.csv')
target_metrics = ['PSNR', 'SSIM', 'FSIM']
baseline_order = ['Poisson 1-NEX', 'Poisson 3-NEX', 'LOUPE 1-NEX', 'LOUPE 2-NEX', 'LOUPE 3-NEX']

def format_cell(row):
    p = row['p_wilcoxon_holm']
    eff = row['median_effect']
    
    p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    
    # Format effect size depending on metric
    if row['metric'] == 'PSNR':
        eff_str = f"+{eff:.2f}dB" if eff > 0 else f"{eff:.2f}dB"
    else:
        eff_str = f"+{eff:.4f}" if eff > 0 else f"{eff:.4f}"
        
    return f"{p_str} ({eff_str})"

for file in sorted(csv_files):
    print(f"\n--- Results for {file} ---")
    df = pd.read_csv(file)
    df = df[df['metric'].isin(target_metrics)]
    
    if df.empty: continue

    df['formatted'] = df.apply(format_cell, axis=1)
    pivot_table = df.pivot(index='metric', columns='baseline', values='formatted')
    
    available_baselines = [b for b in baseline_order if b in pivot_table.columns]
    pivot_table = pivot_table[available_baselines]
    pivot_table = pivot_table.reindex(target_metrics).dropna(how='all')
    
    print(pivot_table.to_markdown())