#!/usr/bin/env bash
# Запуск всех 3 симуляций для San Francisco.
# Из корня проекта: ./run_san_francisco.sh

set -e
cd "$(dirname "$0")"

export DATASET_DIR=dataset/san_francisco
export RESULTS_DIR=results/san_francisco
export SETUP_DIR=setup/san_francisco

echo "=== San Francisco ==="
echo "  simulation_traditional_taxi.py ..."
python simluations/simulation_traditional_taxi.py
echo "  hailing_platform_simulation.py ..."
python simluations/hailing_platform_simulation.py
echo "  simulation_sav.py ..."
python simluations/simulation_sav.py
echo "=== San Francisco: done ==="
