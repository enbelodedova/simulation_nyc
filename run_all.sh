#!/usr/bin/env bash
# Запуск всех 3 симуляций для всех городов: San Francisco → Chicago → DC.
# Из корня проекта: ./run_all.sh

set -e
cd "$(dirname "$0")"

echo "Прогон симуляций: SF → Chicago → DC (trad, hailing, sav)"
echo ""

./run_san_francisco.sh
./run_chicago.sh
./run_dc.sh

echo ""
echo "--- Все прогоны завершены ---"
