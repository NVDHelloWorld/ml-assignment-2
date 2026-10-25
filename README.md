# ML Assignment 2: Decision Trees, Bagging, Boosting & Bias-Variance Analysis on Olivetti Faces

**Author:** Navdeep Doriya  
**Roll Number:** M26MER005  
**Random Seed:** 26005 (derived deterministically from roll number numeric characters)  
**Repository:** [https://github.com/NVDHelloWorld/ml-assignment-2](https://github.com/NVDHelloWorld/ml-assignment-2)  

---

## 1. Overview & Objectives

This repository contains the complete implementation, experimental pipelines, configurations, saved splits, logs, and figure generation scripts for **Machine Learning Assignment 2**.

The study investigates non-linear tree-based classification and ensemble learning on the high-dimensional **Olivetti Faces** dataset (40 identities, 10 images each, $64 \times 64$ raw flattened grayscale pixels $= 4096$ dimensions). We evaluate:
1. **Decision Tree Baseline**: Exhaustive grid search over depth constraints and leaf sample thresholds.
2. **Multiclass AdaBoost (SAMME) from Scratch**: Theoretical derivation and implementation of the SAMME multiclass boosting algorithm, learning rate dynamics, sample-weight trajectories, and identification of the most difficult training identity.
3. **Bagging from Scratch**: Bootstrap aggregation with majority voting, empirical convergence as a function of ensemble size $B \in [1, 200]$, and head-to-head comparison against boosting at matched ensemble sizes ($B \in \{25, 50, 100\}$).
4. **Occlusion Robustness**: Performance degradation under spatial perturbation where a central $16 \times 16$ block (rows 24 to 39, columns 24 to 39) is set to zero on the test set.
5. **Bias-Variance Decomposition**: Implementation of the Kohavi-Wolpert / Breiman decomposition framework across $R=30$ stratified bootstrap refits, evaluating architectural depth scaling (Shallow, Selected, Deep) and quantifying empirical variance suppression.

---

## 2. Key Empirical Findings

| Model Architecture | Hyperparameters | Clean Test Acc (%) | Masked Test Acc (%) | Accuracy Drop (pp) | $\text{Bias}^2$ | Variance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Single Decision Tree** | `max_depth=None, min_leaf=1` | 46.25% | 46.25% | **0.00** | 0.3820 | 0.6614 |
| **Bagging Ensemble** | `B=200` (trees: `max_depth=None`) | **88.75%** | **80.00%** | **8.75** | 0.5096 | **0.0236** |
| **Boosting (SAMME)** | `depth=4, rounds=50` | 36.25% | 22.50% | **13.75** | 0.8880 | 0.0213 |

* **Variance Suppression:** Bagging suppresses estimation variance by **96.4%** relative to an individual unconstrained decision tree ($0.6614 \to 0.0236$), elevating clean test accuracy from $46.25\%$ to $88.75\%$.
* **Robustness vs. Vulnerability:** Under central $16 \times 16$ facial occlusion, Bagging maintains high identification accuracy ($80.00\%$), while Boosting collapses by $13.75$ percentage points to $22.50\%$. Single Decision Trees display artificial stability ($46.25\% \to 46.25\%$) because their axis-aligned splits rely on peripheral pixels outside the occluded region.

---

## 3. Repository Architecture

```text
ml_assignment_2/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Exact Python package dependencies
├── environment.yml                # Conda environment definition
├── reproduce.sh                   # One-click reproduction script (Linux/macOS/Git Bash)
├── run_experiments.py             # Master end-to-end Python runner
├── ML_assignment_2_.ipynb         # Complete executed Google Colab / Jupyter notebook
├── configs/
│   └── config.json                # Complete experimental hyperparameter specifications
├── data/
│   └── olivetti_face_split.npz    # Saved deterministic dataset split (Train: 240, Val: 80, Test: 80)
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading, stratified partitioning & occlusion masking
│   ├── decision_tree.py           # Question 1: Decision tree grid search & leaf analysis
│   ├── boosting.py                # Question 2: SAMME AdaBoost from scratch & toy verification
│   ├── bagging.py                 # Question 2: Bagging classifier from scratch & ensemble search
│   ├── robustness.py              # Question 2: Robustness evaluation & divergent sample extraction
│   └── bias_variance.py           # Question 3: 30-trial bootstrap bias-variance & depth scaling
├── scripts/
│   ├── generate_figures.py        # Publication-grade vector/raster plot generator
│   ├── extract_nb_artifacts.py    # Direct artifact extractor from notebook
│   ├── build_results_from_notebook.py # Verification and artifact builder
│   ├── generate_paper_pdf.py      # CVPR 2-column PDF compiler using Typst
│   └── setup_git_repo.py          # Git repository initialization and commit tracker
├── results/
│   ├── results.json               # Master hierarchical experiment output JSON
│   ├── final_summary_table.csv    # Final comparison table (Accuracy, Drop, Bias, Variance)
│   ├── q1_tree_results.csv        # Decision tree hyperparameter tuning table
│   ├── q2_boosting_grid.csv       # Boosting depth vs. rounds grid search metrics
│   ├── q2_comparison_rows.csv     # Matched Bagging vs. Boosting comparison (B in [25, 50, 100])
│   ├── q2_robustness_eval.csv     # Clean vs. Masked test performance and confidence
│   ├── q3_bias_variance.csv       # Measured Bias^2 and Variance across models
│   ├── q3_depth_analysis.csv      # Depth scaling study metrics (Shallow, Selected, Deep)
│   ├── q3_leaf_analysis.csv       # Tree leaf count and singleton leaf scaling
│   └── commit_hash.txt            # Final commit SHA-1 hash for submission
├── figures/
│   ├── fig1_olivetti_samples.png  # Dataset subject identities
│   ├── fig2_boosting_dynamics.png # SAMME error, alpha, max weight & top-10 weight curves
│   ├── fig3_difficult_sample.png  # Heaviest training face image & round prediction history
│   ├── fig4_masking_comparison.png# Occlusion comparison & accuracy drop chart
│   ├── fig5_bias_variance_decomposition.png # Bias^2 vs. Variance bar chart
│   └── fig6_depth_variance_reduction.png    # Depth-dependent variance suppression
└── paper/
    ├── cvpr_paper.typ             # Professional Typst CVPR two-column source
    └── CVPR_Paper_M26MER005.pdf   # Compiled CVPR format report (max 8 pages excluding references)
```

---

## 4. Exact Reproduction Commands

### Option A: Bash Script (Recommended)
```bash
chmod +x reproduce.sh
./reproduce.sh
```

### Option B: Manual Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end experiment pipeline
python run_experiments.py

# 3. Compile the CVPR two-column research paper PDF
python scripts/generate_paper_pdf.py
```

---

## 5. Algorithmic Formulations

### Multiclass SAMME Boosting (From Scratch)
For $K=40$ classes, the multi-class Stagewise Additive Modeling using a Multi-class Exponential loss function (SAMME) proceeds as:
1. Initialize uniform sample weights: $w_1(i) = \frac{1}{N}, \forall i=1, \dots, N$.
2. For each round $t = 1, \dots, T$:
   * Fit weak learner $h_t(x)$ using distribution $w_t$.
   * Compute weighted training error:
     $$e_t = \frac{\sum_{i=1}^N w_t(i) \cdot \mathbb{I}(h_t(x_i) \neq y_i)}{\sum_{i=1}^N w_t(i)}$$
   * Reject learner if $e_t \ge 1 - \frac{1}{K} = \frac{39}{40} = 0.975$.
   * Compute estimator weight:
     $$\alpha_t = \ln \left( \frac{1 - e_t}{e_t} \right) + \ln(K - 1)$$
   * Update and normalize sample weights:
     $$w_{t+1}(i) \propto w_t(i) \exp \left( \alpha_t \cdot \mathbb{I}(h_t(x_i) \neq y_i) \right)$$
3. Aggregated decision rule:
   $$H(x) = \arg\max_{k \in \{0, \dots, K-1\}} \sum_{t=1}^T \alpha_t \cdot \mathbb{I}(h_t(x) = k)$$

### Bias-Variance Decomposition (Kohavi-Wolpert / Breiman)
Across $R=30$ independent stratified bootstrap draws from the training set on $N=80$ fixed test examples:
$$\text{Bias}^2 = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \left( \bar{p}_k(x_i) - y_{ik}^* \right)^2, \quad \text{where } \bar{p}_k(x_i) = \frac{1}{R} \sum_{r=1}^R p_{r,k}(x_i)$$
$$\text{Variance} = \frac{1}{N} \sum_{i=1}^N \frac{1}{R} \sum_{r=1}^R \sum_{k=1}^K \left( p_{r,k}(x_i) - \bar{p}_k(x_i) \right)^2$$
