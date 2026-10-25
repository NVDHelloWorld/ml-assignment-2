#!/usr/bin/env bash
# ==============================================================================
# Exact Reproduction Script for ML Assignment 2
# Student: Navdeep Doriya | Roll Number: M26MER005 | Random Seed: 26005
# ==============================================================================

set -e

echo "=== Step 1: Environment Setup ==="
python -m pip install -r requirements.txt

echo "=== Step 2: Running End-to-End Experiments ==="
python run_experiments.py | tee results/execution.log
echo "=== Reproduction Complete! ==="

