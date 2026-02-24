#!/usr/bin/env bash
# Запуск всех 3 симуляций для DC.
# Из корня проекта: ./run_dc.sh

set -e
cd "$(dirname "$0")"

export DATASET_DIR=dataset/dc
export RESULTS_DIR=results/dc
export SETUP_DIR=setup/dc

echo "=== DC ==="
echo "  simulation_traditional_taxi.py ..."
python simluations/simulation_traditional_taxi.py
echo "  hailing_platform_simulation.py ..."
python simluations/hailing_platform_simulation.py
echo "  simulation_sav.py ..."
python simluations/simulation_sav.py
echo "=== DC: done ==="
