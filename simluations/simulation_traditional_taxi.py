from datetime import datetime
from simulation_functions import greedy_two_pass_assignment, calculate_on_line_cars
from paths_config import MEAN_DISTANCE_TIME_PATH, DAILY_TRIPS_DIR, RESULTS_DIR, SETUP_DIR
import uuid
import os
from pathlib import Path

import pandas as pd

import random

max_wait_time = 900
co2_per_km = 4.8

mean_distance_time = pd.read_csv(MEAN_DISTANCE_TIME_PATH)
unique_locations = pd.concat([mean_distance_time['PULocationID'], mean_distance_time['DOLocationID']]).unique()

start_hour = 6
end_hour = 21


def assign_rides(current_trips, trips_data_day, current_time):
    """Assign cars to trips based on proximity and wait time (optimized vectorized version)."""
    if current_trips.empty or fleet[fleet["status"] == "idle"].empty:
        return

    current_trips = current_trips.copy()

    available_cars = fleet[fleet["status"] == "idle"].copy()

    current_trips['_key'] = 1
    available_cars['_key'] = 1
    combinations = pd.merge(current_trips, available_cars, on='_key').drop('_key', axis=1)

    combinations = combinations.merge(
        mean_distance_time,
        left_on=['location_id', 'PULocationID'],
        right_on=['PULocationID', 'DOLocationID'],
        how='inner',
        suffixes=['', '_est']
    ).rename(columns={
        'trip_time_est': 'est_time_to_pickup',
        'trip_miles_est': 'est_miles_to_pickup'
    })

    best_assignments = greedy_two_pass_assignment(combinations)
    best_assignments['eta_to_pickup'] = current_time + pd.to_timedelta(best_assignments['est_time_to_pickup'], unit='s')
    best_assignments['eta_to_dropoff'] = best_assignments['eta_to_pickup'] + pd.to_timedelta(
        best_assignments['trip_time'], unit='s')

    car_id_to_index = dict(zip(fleet['car_id'], fleet.index))
    assign_indices = best_assignments['car_id'].map(car_id_to_index)

    fleet_update_df = pd.DataFrame({
        'index': assign_indices,
        'current_trip_id': best_assignments['trip_id'].values,
        'eta_to_pickup': best_assignments['eta_to_pickup'].values,
        'eta_to_dropoff': best_assignments['eta_to_dropoff'].values
    })

    fleet.loc[fleet_update_df['index'], 'status'] = 'en_route_to_client'
    fleet.loc[fleet_update_df['index'], 'current_trip_id'] = fleet_update_df['current_trip_id'].values
    fleet.loc[fleet_update_df['index'], 'eta_to_pickup'] = fleet_update_df['eta_to_pickup'].values
    fleet.loc[fleet_update_df['index'], 'eta_to_dropoff'] = fleet_update_df['eta_to_dropoff'].values

    trips_data_day.loc[trips_data_day['trip_id'].isin(best_assignments['trip_id']), 'assigned'] = True

    trip_id_to_car = dict(zip(best_assignments['trip_id'], best_assignments['car_id']))
    trips_data_day.loc[trips_data_day['trip_id'].isin(trip_id_to_car.keys()), 'completed_by'] = \
        trips_data_day['trip_id'].map(trip_id_to_car)


def update_car_status(trips_data, current_time):
    if (fleet["status"] == "en_route_to_client").any():
        mask_to_client = (fleet["status"] == "en_route_to_client") & (fleet["eta_to_pickup"] <= current_time)
        fleet.loc[mask_to_client, "status"] = "en_route_with_client"
        trip_ids = fleet.loc[mask_to_client, "current_trip_id"]
        trips_data.loc[trips_data['trip_id'].isin(trip_ids), 'actual_eta_pickup'] = current_time

    mask_with_client = (fleet["status"] == "en_route_with_client") & (
            pd.to_datetime(fleet["eta_to_dropoff"]) <= current_time)
    if mask_with_client.any():
        completed_fleet = fleet.loc[mask_with_client]
        fleet.loc[mask_with_client, "status"] = "idle"
        fleet.loc[mask_with_client, "current_trip_id"] = None
        trip_info = trips_data.set_index("trip_id").loc[completed_fleet["current_trip_id"]][["DOLocationID"]]
        fleet.loc[mask_with_client, "location_id"] = trip_info.values
        trips_data.loc[trips_data['trip_id'].isin(completed_fleet["current_trip_id"]), 'complete_time'] = current_time


def run_daily_simulation(trips_data_day, start_time, fleet, minutes_in_day=1441):
    on_line_cars = pd.DataFrame()

    for t in range(minutes_in_day):
        current_time = start_time + pd.to_timedelta(t, unit="m")

        current_trips = trips_data_day[
            (trips_data_day["assigned"] == False) &
            ((current_time - trips_data_day['request_datetime']).dt.total_seconds() < max_wait_time) &
            ((current_time - trips_data_day['request_datetime']).dt.total_seconds() >= 0) &
            (trips_data_day['request_datetime'].dt.hour >= start_hour) &
            (trips_data_day['request_datetime'].dt.hour < end_hour)
            ]

        if len(fleet[fleet['status'] == 'idle']) > 0:
            assign_rides(current_trips, trips_data_day, current_time)

        update_car_status(trips_data_day, current_time)
        on_line_cars = calculate_on_line_cars(fleet, on_line_cars, current_time)
        print('simulated', current_time)

    completed_trips = trips_data_day[(trips_data_day['complete_time'] <= current_time) & \
                                     (trips_data_day['complete_time'] > pd.to_datetime('1999-01-01'))
                                     ]
    current_trips = pd.concat([trips_data_day[
                                   (~trips_data_day["assigned"]) &
                                   ((current_time - trips_data_day[
                                       'request_datetime']).dt.total_seconds() < max_wait_time) &
                                   ((current_time - trips_data_day['request_datetime']).dt.total_seconds() >= 0)
                                   ],
                               trips_data_day[(trips_data_day['complete_time'] == pd.to_datetime('1999-01-01')) & \
                                              (trips_data_day['assigned'])
                                              ]
                               ])
    lost_trips = trips_data_day[(~trips_data_day["assigned"]) & \
                                ((current_time - trips_data_day[
                                    'request_datetime']).dt.total_seconds() >= max_wait_time)]

    return completed_trips, current_trips, lost_trips, on_line_cars, fleet


fleet_sizes = [100, 500, 1000, 2000, 3500]

for i in range(len(fleet_sizes)):
    fleet_size = fleet_sizes[i]

    fleet = pd.DataFrame({
        "location_id": random.choices(unique_locations, k=fleet_size),
        "status": ["idle"] * fleet_size,  # idle, en_route_to_client, en_route_with_client
        "current_trip_id": [None] * fleet_size,
        "car_id": [str(uuid.uuid4()) for _ in range(fleet_size)],
        "eta_to_dropoff": [None] * fleet_size
    })

    current_trips = pd.DataFrame()

    folder_path = DAILY_TRIPS_DIR
    file_paths = sorted(
        [folder_path / f for f in os.listdir(folder_path) if (folder_path / f).is_file()]
    )

    for file_path in file_paths:
        print(f"Running simulation for day {file_path}")
        day_trips = pd.concat([current_trips, pd.read_parquet(str(file_path), engine='fastparquet')], ignore_index=True)
        day_trips['request_datetime'] = pd.to_datetime(day_trips['request_datetime'])
        day_trips = day_trips.sort_values('request_datetime').reset_index(drop=True)
        # start_time по дате текущего файла (trips_YYYY-MM-DD), иначе при переносе current_trips
        # с прошлого дня min(day_trips) = старый день и новые поездки не попадают в окно
        file_path = Path(file_path)
        day_date_str = file_path.stem.replace('trips_', '')
        start_time = pd.Timestamp(day_date_str).normalize()

        completed_trips, current_trips, lost_trips, online_cars, fleet = run_daily_simulation(
            day_trips,
            start_time,
            fleet
        )

        final_completed_trips = pd.concat([completed_trips, lost_trips], axis=0)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        SETUP_DIR.mkdir(parents=True, exist_ok=True)
        completed_path = RESULTS_DIR / f"trad_completed_trips_day_{fleet_size}.parquet"
        online_path = RESULTS_DIR / f"trad_online_cars_day_{fleet_size}.parquet"

        if completed_path.exists():
            existing = pd.read_parquet(str(completed_path), engine='pyarrow')
            final_completed_trips = pd.concat([existing, final_completed_trips], ignore_index=True)
            existing_online = pd.read_parquet(str(online_path), engine='pyarrow')
            online_cars = pd.concat([existing_online, online_cars], ignore_index=True)
        final_completed_trips.to_parquet(str(completed_path), engine='pyarrow', index=False)
        online_cars.to_parquet(str(online_path), engine='pyarrow', index=False)

        fleet.to_csv(str(SETUP_DIR / f"trad_fleet_final_{fleet_size}.csv"), index=False)
