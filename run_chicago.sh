#!/usr/bin/env bash
# Запуск всех 3 симуляций для Chicago.
# Из корня проекта: ./run_chicago.sh

set -e
cd "$(dirname "$0")"

export DATASET_DIR=dataset/chicago
export RESULTS_DIR=results/chicago
export SETUP_DIR=setup/chicago

echo "=== Chicago ==="
echo "  simulation_traditional_taxi.py ..."
python simluations/simulation_traditional_taxi.py
echo "  hailing_platform_simulation.py ..."
python simluations/hailing_platform_simulation.py
echo "  simulation_sav.py ..."
python simluations/simulation_sav.py
echo "=== Chicago: done ==="
