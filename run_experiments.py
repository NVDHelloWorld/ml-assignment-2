"""
End-to-end execution of all experiments for ML Assignment 2.
Author: Navdeep Doriya (M26MER005)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.dataset import load_and_split_olivetti, create_masked_dataset, ROLL_NUMBER, extract_seed
from src.decision_tree import run_decision_tree_grid_search, analyze_tree_leaves
from src.boosting import SAMMEFromScratch, verify_toy_example, run_boosting_grid_search
from src.bagging import BaggingFromScratch, run_bagging_grid_search
from src.robustness import evaluate_robustness, find_divergent_test_sample
from src.bias_variance import (
    run_bias_variance_study,
    run_depth_bias_variance_study,
    run_leaf_analysis
)
from scripts.generate_figures import (
    plot_olivetti_samples,
    plot_boosting_dynamics,
    plot_difficult_sample,
    plot_masking_comparison,
    plot_bias_variance_decomposition,
    plot_depth_variance_reduction
)

def main():
    print("=" * 70)
    print("ML Assignment 2: Decision Trees, Bagging, Boosting & Robustness")
    print(f"Student: Navdeep Doriya | Roll Number: {ROLL_NUMBER}")
    print("=" * 70)
    
    seed = extract_seed(ROLL_NUMBER)
    print(f"Extracted numeric seed: {seed}")
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # -------------------------------------------------------------
    # Step 1: Load and Split Olivetti Faces
    # -------------------------------------------------------------
    print("\n--- Step 1: Loading & Splitting Olivetti Faces ---")
    data_dict = load_and_split_olivetti(save_path="data/olivetti_face_split.npz", roll_number=ROLL_NUMBER)
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
    plot_olivetti_samples(data_dict["images"], data_dict["y"], "figures/fig1_olivetti_samples.png")
    
    # -------------------------------------------------------------
    # Step 2: Question 1 - Decision Tree Baseline & Tuning
    # -------------------------------------------------------------
    print("\n--- Step 2: Question 1 - Decision Tree Grid Search ---")
    dt_res = run_decision_tree_grid_search(X_train, y_train, X_val, y_val, seed=seed)
    best_tree_key = dt_res["best_key"]
    best_tree = dt_res["best_model"]
    dt_res["results_df"].to_csv("results/q1_tree_results.csv", index=False)
    
    tree_train_acc = best_tree.score(X_train, y_train)
    tree_val_acc = best_tree.score(X_val, y_val)
    tree_test_acc = best_tree.score(X_test, y_test)
    tree_leaf_stats = analyze_tree_leaves(best_tree, X_train)
    
    print(f"Selected Tree: max_depth={best_tree_key[0]}, min_samples_leaf={best_tree_key[1]}")
    print(f"Train Acc: {tree_train_acc:.4f}, Val Acc: {tree_val_acc:.4f}, Test Acc: {tree_test_acc:.4f}")
    print(f"Leaves: {tree_leaf_stats['number_of_leaves']}, Mean samples/leaf: {tree_leaf_stats['mean_samples_per_leaf']:.2f}, Singleton fraction: {tree_leaf_stats['fraction_singleton_leaves']:.4f}")
    
    # -------------------------------------------------------------
    # Step 3: Question 2 - Boosting (SAMME From Scratch)
    # -------------------------------------------------------------
    print("\n--- Step 3: Question 2 - SAMME Boosting From Scratch ---")
    toy_res = verify_toy_example()
    print(f"Toy verification: Error={toy_res['weighted_error']:.4f}, Threshold={toy_res['random_threshold']:.4f}, Alpha={toy_res['alpha']:.4f}")
    
    boost_res = run_boosting_grid_search(X_train, y_train, X_val, y_val, seed=seed)
    best_boost_key = boost_res["best_key"]
    best_booster = boost_res["best_model"]
    boost_res["results_df"].to_csv("results/q2_boosting_grid.csv", index=False)
    
    boost_train_acc = best_booster.score(X_train, y_train)
    boost_val_acc = best_booster.score(X_val, y_val)
    boost_test_acc = best_booster.score(X_test, y_test)
    
    print(f"Selected Boosting: Depth={best_boost_key[0]}, Rounds={best_boost_key[1]}")
    print(f"Train Acc: {boost_train_acc:.4f}, Val Acc: {boost_val_acc:.4f}, Test Acc: {boost_test_acc:.4f}")
    
    plot_boosting_dynamics(best_booster, "figures/fig2_boosting_dynamics.png")
    
    # Find most difficult training face
    final_weights = best_booster.weight_history_[-1]
    difficult_idx = int(np.argmax(final_weights))
    print(f"Most difficult face in training: Index #{difficult_idx} (Class {y_train[difficult_idx]}), Final Weight={final_weights[difficult_idx]:.5f}")
    plot_difficult_sample(X_train, y_train, difficult_idx, best_booster, "figures/fig3_difficult_sample.png")
    
    # -------------------------------------------------------------
    # Step 4: Question 2 - Bagging From Scratch & Comparison
    # -------------------------------------------------------------
    print("\n--- Step 4: Question 2 - Bagging From Scratch & Grid Search ---")
    tree_template = DecisionTreeClassifier(
        max_depth=best_tree_key[0],
        min_samples_leaf=best_tree_key[1],
        random_state=seed
    )
    
    # Comparison table for B in [25, 50, 100]
    comparison_rows = []
    comp_bag_models = {}
    comp_boost_models = {}
    
    for B in [25, 50, 100]:
        # Bagging
        bag = BaggingFromScratch(tree_template=tree_template, n_estimators=B, random_state=seed)
        t0 = time.perf_counter()
        bag.fit(X_train, y_train)
        t_bag = time.perf_counter() - t0
        comp_bag_models[B] = bag
        comparison_rows.append({
            "Method": "Bagging",
            "Learners": B,
            "Train Accuracy": float(bag.score(X_train, y_train)),
            "Validation Accuracy": float(bag.score(X_val, y_val)),
            "Clean Test Accuracy": float(bag.score(X_test, y_test)),
            "Training Time (s)": float(t_bag)
        })
        
        # Boosting
        boost = SAMMEFromScratch(max_depth=best_boost_key[0], n_estimators=B, random_state=seed)
        t0 = time.perf_counter()
        boost.fit(X_train, y_train)
        t_boost = time.perf_counter() - t0
        comp_boost_models[B] = boost
        comparison_rows.append({
            "Method": "Boosting",
            "Learners": B,
            "Train Accuracy": float(boost.score(X_train, y_train)),
            "Validation Accuracy": float(boost.score(X_val, y_val)),
            "Clean Test Accuracy": float(boost.score(X_test, y_test)),
            "Training Time (s)": float(t_boost)
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("results/q2_comparison_rows.csv", index=False)
    print(comp_df.to_string(index=False))
    
    # Full Bagging Grid Search
    bag_res = run_bagging_grid_search(X_train, y_train, X_val, y_val, tree_template, seed=seed)
    final_B = bag_res["best_B"]
    final_bagger = bag_res["best_model"]
    print(f"Selected Bagging Ensemble Size: B={final_B}, Val Acc={final_bagger.score(X_val, y_val):.4f}")
    
    # -------------------------------------------------------------
    # Step 5: Question 2 - Robustness Occlusion Experiment
    # -------------------------------------------------------------
    print("\n--- Step 5: Question 2 - Robustness Occlusion Experiment ---")
    X_test_masked = create_masked_dataset(X_test)
    
    models_dict = {
        "Single tree": (best_tree, lambda X: best_tree.predict_proba(X).max(axis=1)),
        "Bagging": (final_bagger, lambda X: final_bagger.predict_confidence(X)),
        "Boosting": (best_booster, lambda X: best_booster.predict_proba(X).max(axis=1))
    }
    
    rob_res = evaluate_robustness(models_dict, X_test, X_test_masked, y_test)
    rob_summary_df = rob_res["summary_df"]
    rob_summary_df.to_csv("results/q2_robustness_eval.csv", index=False)
    print(rob_summary_df.to_string(index=False))
    
    div_sample = find_divergent_test_sample(best_tree, final_bagger, best_booster, X_test, X_test_masked, y_test)
    print(f"Divergent test sample index: #{div_sample['test_index']} (True ID {div_sample['true_identity']})")
    print(f"Bagging: Clean={div_sample['bag_clean']}, Masked={div_sample['bag_masked']}")
    print(f"Boosting: Clean={div_sample['boost_clean']}, Masked={div_sample['boost_masked']}")
    plot_masking_comparison(X_test, X_test_masked, div_sample["test_index"], y_test, rob_summary_df, "figures/fig4_masking_comparison.png")
    
    # -------------------------------------------------------------
    # Step 6: Question 3 - Bias-Variance Study (R=30 Repetitions)
    # -------------------------------------------------------------
    print("\n--- Step 6: Question 3 - Bias-Variance Study (R=30 Repetitions) ---")
    selected_models_config = {
        "Single tree": {"depth": best_tree_key[0], "min_leaf": best_tree_key[1]},
        "Bagging": {"depth": best_tree_key[0], "min_leaf": best_tree_key[1], "B": final_B},
        "Boosting": {"depth": best_boost_key[0], "rounds": best_boost_key[1]}
    }
    
    bv_res = run_bias_variance_study(X_train, y_train, X_test, y_test, selected_models_config, R=30, seed=seed)
    bv_summary_df = bv_res["summary_df"]
    bv_summary_df.to_csv("results/q3_bias_variance.csv", index=False)
    print(bv_summary_df.to_string(index=False))
    plot_bias_variance_decomposition(bv_summary_df, "figures/fig5_bias_variance_decomposition.png")
    
    # -------------------------------------------------------------
    # Step 7: Question 3 - Depth Scaling & Leaf Analysis
    # -------------------------------------------------------------
    print("\n--- Step 7: Question 3 - Depth Scaling & Leaf Analysis ---")
    depth_res = run_depth_bias_variance_study(X_train, y_train, X_test, y_test, best_tree_key[0], R=30, seed=seed)
    depth_df = depth_res["depth_df"]
    depth_df.to_csv("results/q3_depth_analysis.csv", index=False)
    print(depth_df.to_string(index=False))
    plot_depth_variance_reduction(depth_df, "figures/fig6_depth_variance_reduction.png")
    
    leaf_df = run_leaf_analysis(X_train, y_train, depth_res["depth_settings"], R=30, seed=seed)
    leaf_df.to_csv("results/q3_leaf_analysis.csv", index=False)
    print(leaf_df.to_string(index=False))
    
    # -------------------------------------------------------------
    # Step 8: Master Final Comparison Table
    # -------------------------------------------------------------
    print("\n--- Step 8: Master Summary Table ---")
    final_rows = []
    for model_name in ["Single tree", "Bagging", "Boosting"]:
        rob_row = rob_summary_df[rob_summary_df["Model"] == model_name].iloc[0]
        bv_row = bv_summary_df[bv_summary_df["Model"] == model_name].iloc[0]
        
        final_rows.append({
            "Model": model_name,
            "Clean Test Accuracy": float(rob_row["Clean Accuracy"]),
            "Masked Test Accuracy": float(rob_row["Masked Accuracy"]),
            "Accuracy Drop (pp)": float(rob_row["Accuracy Drop (pp)"]),
            "Bias Squared": float(bv_row["Bias squared"]),
            "Variance": float(bv_row["Variance"])
        })
        
    master_table = pd.DataFrame(final_rows)
    master_table.to_csv("results/final_summary_table.csv", index=False)
    print(master_table.to_string(index=False))
    
    # Save complete JSON results
    json_results = {
        "student": {
            "name": "Navdeep Doriya",
            "roll_number": ROLL_NUMBER,
            "seed": seed
        },
        "question_1": {
            "best_max_depth": str(best_tree_key[0]),
            "best_min_samples_leaf": int(best_tree_key[1]),
            "train_accuracy": float(tree_train_acc),
            "val_accuracy": float(tree_val_acc),
            "test_accuracy": float(tree_test_acc),
            "leaf_statistics": tree_leaf_stats
        },
        "question_2": {
            "toy_verification": {
                "error": float(toy_res["weighted_error"]),
                "threshold": float(toy_res["random_threshold"]),
                "alpha": float(toy_res["alpha"])
            },
            "boosting": {
                "selected_depth": int(best_boost_key[0]),
                "selected_rounds": int(best_boost_key[1]),
                "train_accuracy": float(boost_train_acc),
                "val_accuracy": float(boost_val_acc),
                "test_accuracy": float(boost_test_acc),
                "difficult_sample": {
                    "train_idx": difficult_idx,
                    "class": int(y_train[difficult_idx]),
                    "final_weight": float(final_weights[difficult_idx])
                }
            },
            "bagging": {
                "selected_B": int(final_B),
                "val_accuracy": float(final_bagger.score(X_val, y_val)),
                "clean_test_accuracy": float(final_bagger.score(X_test, y_test))
            },
            "robustness": rob_summary_df.to_dict(orient="records"),
            "divergent_sample": div_sample
        },
        "question_3": {
            "bias_variance": bv_summary_df.to_dict(orient="records"),
            "depth_scaling": depth_df.to_dict(orient="records"),
            "leaf_scaling": leaf_df.to_dict(orient="records")
        },
        "master_summary": master_table.to_dict(orient="records")
    }
    
    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
        
    print("\nAll experiments completed and artifacts generated successfully!")

if __name__ == "__main__":
    main()
