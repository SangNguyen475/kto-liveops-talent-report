#!/usr/bin/env bash
# Run the whole pipeline and save the console output to results/run_log.txt
set -e
cd "$(dirname "$0")/src"
{
  echo "== 1. Generate the sample";      python generate_data.py
  echo; echo "== 2. Train the models";   python train_model.py
  echo; echo "== 3. Sample size";        python sample_size.py
  echo; echo "== 4. Dashboard mock-up";  python dashboard_mockup.py
} | tee ../results/run_log.txt
