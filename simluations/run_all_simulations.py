import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMLUATIONS_DIR = Path(__file__).resolve().parent

CITIES = ["san_francisco", "chicago", "dc"]
SCRIPTS = [
    "simulation_traditional_taxi.py",
    "hailing_platform_simulation.py",
    "simulation_sav.py",
]


def run_simulation(city: str, script: str) -> bool:
    env = os.environ.copy()
    env["DATASET_DIR"] = str(PROJECT_ROOT / "dataset" / city)
    env["RESULTS_DIR"] = str(PROJECT_ROOT / "results" / city)
    env["SETUP_DIR"] = str(PROJECT_ROOT / "setup" / city)
    script_path = SIMLUATIONS_DIR / script
    if not script_path.exists():
        print(f"  [SKIP] {script} не найден")
        return False
    print(f"  Запуск {script} ...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=None,
        )
        ok = result.returncode == 0
        if ok:
            print(f"  [OK] {script}")
        else:
            print(f"  [FAIL] {script} exit code {result.returncode}")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {script}")
        return False
    except Exception as e:
        print(f"  [ERROR] {script}: {e}")
        return False


def main():
    print("Прогон симуляций: города SF → Chicago → DC, скрипты trad → hailing → sav\n")
    results = []
    for city in CITIES:
        dataset_dir = PROJECT_ROOT / "dataset" / city
        daily_dir = dataset_dir / "daily_trips"
        if not daily_dir.exists() or not list(daily_dir.glob("trips_*.parquet")):
            print(f"[SKIP] {city}: нет daily_trips с trips_*.parquet")
            results.append((city, None, "skip"))
            continue
        print(f"=== {city} ===")
        for script in SCRIPTS:
            ok = run_simulation(city, script)
            results.append((city, script, "ok" if ok else "fail"))
        print()
    # сводка
    print("--- Сводка ---")
    for city, script, status in results:
        if script is None:
            print(f"  {city}: {status}")
        else:
            print(f"  {city} / {script}: {status}")
    fails = sum(1 for _, s, st in results if s and st == "fail")
    if fails:
        sys.exit(1)
    print("\nВсе прогоны завершены успешно.")


if __name__ == "__main__":
    main()
