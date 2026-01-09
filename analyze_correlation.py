import polars as pl
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import gc

def load_style_embeddings(path, device):
    """
    Loads style embeddings into a Dictionary {author: tensor} on GPU.
    Normalizes them immediately for fast Cosine Similarity.
    """
    print(f"Loading Style Embeddings from {path}...")
    df = pl.read_parquet(path)
    
    style_map = {}
    
    batch_size = 10000
    for i in range(0, len(df), batch_size):
        chunk = df.slice(i, batch_size)
        authors = chunk["author"].to_list()
        # Cast to float32 to save memory and match model output
        embeddings = torch.tensor(chunk["embedding"].to_list(), dtype=torch.float32, device=device)
        
        # Robust NaN handling
        if torch.isnan(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0)

        # Normalize L2
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        
        for j, auth in enumerate(authors):
            style_map[auth] = embeddings[j]
            
    print(f"-> Loaded {len(style_map)} style vectors.")
    return style_map

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # 2D dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def main():
    parser = argparse.ArgumentParser(description="Correlate Semantics vs Style")
    parser.add_argument("answer_file", help="Path to answer_embeddings.parquet")
    parser.add_argument("style_file", help="Path to author_style_embeddings.parquet")
    parser.add_argument("--max-pairs", type=int, default=2_000_000, help="Stop after collecting this many pairs")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Styles
    style_map = load_style_embeddings(args.style_file, device)

    # 2. Process Answers
    print(f"Scanning Answer Embeddings from {args.answer_file}...")
    
    sem_sims_all = []
    sty_sims_all = []
    total_pairs_collected = 0
    
    try:
        df_answers = pl.read_parquet(args.answer_file)
    except Exception as e:
        print(f"Error reading answer file: {e}")
        sys.exit(1)

    unique_questions = df_answers["question_id"].unique().to_list()
    print(f"Found {len(unique_questions)} unique questions. Processing...")
    
    grouped = df_answers.partition_by("question_id", as_dict=True)
    
    pbar = tqdm(total=args.max_pairs, desc="Collecting Pairs", unit="pair")
    
    for q_id, group in grouped.items():
        if len(group) < 2:
            continue 
        
        authors = group["author"].to_list()
        
        # Filter: Ensure we have style embeddings
        valid_indices = [i for i, a in enumerate(authors) if a in style_map]
        
        if len(valid_indices) < 2:
            continue
            
        raw_sem_embs = group["embedding"].to_list()
        valid_sem_embs = [raw_sem_embs[i] for i in valid_indices]
        valid_authors = [authors[i] for i in valid_indices]
        
        # Stack to Tensor
        t_sem = torch.tensor(valid_sem_embs, dtype=torch.float32, device=device)
        
        if torch.isnan(t_sem).any():
            t_sem = torch.nan_to_num(t_sem, nan=0.0)

        t_sem = F.normalize(t_sem, p=2, dim=1, eps=1e-8)
        
        # Get Style Embeddings
        t_sty = torch.stack([style_map[a] for a in valid_authors])
        
        # Matrices
        sim_matrix_sem = torch.mm(t_sem, t_sem.t())
        sim_matrix_sty = torch.mm(t_sty, t_sty.t())
        
        # Upper Triangle
        rows, cols = torch.triu_indices(len(valid_authors), len(valid_authors), offset=1)
        
        vals_sem = sim_matrix_sem[rows, cols]
        vals_sty = sim_matrix_sty[rows, cols]
        
        sem_sims_all.append(vals_sem.cpu().numpy())
        sty_sims_all.append(vals_sty.cpu().numpy())
        
        count = len(rows)
        total_pairs_collected += count
        pbar.update(count)
        
        if total_pairs_collected >= args.max_pairs:
            break
            
    pbar.close()
    
    if total_pairs_collected == 0:
        print("No valid pairs found.")
        sys.exit(0)

    print("Concatenating data...")
    X = np.concatenate(sem_sims_all)
    y = np.concatenate(sty_sims_all)
    
    del style_map
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # CLEANING
    # ---------------------------------------------------------
    print("\n--- Cleaning Data ---")
    mask = np.isfinite(X) & np.isfinite(y)
    n_removed = len(X) - np.sum(mask)
    
    if n_removed > 0:
        print(f"Removed {n_removed} corrupted pairs (NaN/Inf).")
        X = X[mask]
        y = y[mask]

    if len(X) < 100:
        print("Error: Not enough data points.")
        sys.exit(1)

    # ---------------------------------------------------------
    # STATISTICAL ANALYSIS
    # ---------------------------------------------------------
    print(f"Total Valid Pairs: {len(X):,}")
    
    # 1. Linear Analysis
    corr, p_value = pearsonr(X, y)
    print(f"\n[Linear] Pearson Correlation: {corr:.4f} (p={p_value:.2e})")
    
    X_reshaped = X.reshape(-1, 1)
    linear_reg = LinearRegression().fit(X_reshaped, y)
    r2 = linear_reg.score(X_reshaped, y)
    slope = linear_reg.coef_[0]
    intercept = linear_reg.intercept_
    
    print(f"[Linear] R-squared: {r2:.4f}")

    # ---------------------------------------------------------
    # VISUALIZATION (Marginal Histograms)
    # ---------------------------------------------------------
    print("\nGenerating plot...")
    
    # Setup Grid Spec
    # Layout:
    # [ Y-Hist ] [ Main Plot ]
    # [ Blank  ] [ X-Hist    ]
    
    fig = plt.figure(figsize=(10, 10))
    # width_ratios: Side hist gets 1 part, Main plot gets 4 parts
    gs = fig.add_gridspec(2, 2,  width_ratios=(1, 4), height_ratios=(4, 1),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    # Create Axes
    ax_main = fig.add_subplot(gs[0, 1])
    ax_histx = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[0, 0], sharey=ax_main)

    # --- MAIN SCATTER (HEXBIN) ---
    hb = ax_main.hexbin(X, y, gridsize=60, cmap='inferno', mincnt=1, bins='log')
    
    # Add Colorbar (Inset or external - putting it inside to save space)
    # We create a small axis inside the main plot for the colorbar
    cb_ax = ax_main.inset_axes([0.95, 0.05, 0.02, 0.4]) 
    fig.colorbar(hb, cax=cb_ax, label='log10(Count)')

    # Linear Regression Line
    x_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_linear_pred = linear_reg.predict(x_grid)
    ax_main.plot(x_grid, y_linear_pred, color='cyan', linestyle='--', linewidth=2, label=f'Linear Fit (r={corr:.2f})')

    # Confidence Ellipses
    confidence_ellipse(X, y, ax_main, n_std=1, edgecolor='magenta', linestyle='-', linewidth=2, label='1 Std Dev')
    confidence_ellipse(X, y, ax_main, n_std=2, edgecolor='magenta', linestyle='--', linewidth=1.5, label='2 Std Dev')
    confidence_ellipse(X, y, ax_main, n_std=3, edgecolor='magenta', linestyle=':', linewidth=1, label='3 Std Dev')

    # Centroid
    ax_main.scatter(np.mean(X), np.mean(y), c='magenta', s=50, marker='x')

    # --- HISTOGRAMS ---
    bins = 50
    color = 'gray'
    
    # X Histogram (Bottom)
    ax_histx.hist(X, bins=bins, color=color, alpha=0.7)
    
    # Y Histogram (Left)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color=color, alpha=0.7)
    # Invert X axis of the left plot so bars grow towards the left? 
    # Usually standard is base on the left axis. Let's keep base on left axis.
    ax_histy.invert_xaxis() 

    # --- STYLING ---
    # Hide labels on inner axes
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_main.get_yticklabels(), visible=False)
    
    # Axis Labels on the outer histograms
    ax_histx.set_xlabel('Semantic Similarity (Opinion)')
    ax_histy.set_ylabel('Style Similarity (Writing)')
    
    # Remove Count Ticks (Visual Clutter)
    ax_histx.set_yticks([])
    ax_histy.set_xticks([])

    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title("Correlation Analysis: Semantic vs Style")

    out_file = "correlation_marginal_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
