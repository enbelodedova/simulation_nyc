from datetime import datetime
import uuid
import os
from pathlib import Path
from simulation_functions import greedy_two_pass_assignment, calculate_on_line_cars

import pandas as pd

import random

charging_time = 60 * 5  # time to charge fo a full battery in minutes
distance_per_charge = 300  # range a car that travle on full battery in km
speed_of_charging = distance_per_charge / charging_time

# charging_stations_count = 10
charging_station_capacity = 5
electricity_consumption_per_km = 1  # in units of battery level consumed per km
max_wait_time = 900  # Maximum wait time for trip in seconds (15 minutes)
co2_per_km = 0.4  # Indirect emission through charging

mean_distance_time = pd.read_csv('../dataset/mean_distance_time.csv')
unique_locations = pd.concat([mean_distance_time['PULocationID'], mean_distance_time['DOLocationID']]).unique()


def assign_rides(current_trips, trips_data_day, current_time):
    """Assign cars to trips based on proximity, battery level, and wait time (optimized vectorized version)."""
    if current_trips.empty or fleet[fleet["status"] == "idle"].empty:
        return

    current_trips = current_trips.copy()

    current_trips['required_battery'] = current_trips['trip_miles'] * electricity_consumption_per_km

    available_cars = fleet[fleet["status"] == "idle"].copy()

    # Cross join using merge on dummy key (faster than assign(key=1))
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

    combinations['battery_total_required'] = (
        combinations['required_battery'] +
        combinations['est_miles_to_pickup'] * electricity_consumption_per_km
    )
    valid_combinations = combinations[combinations['battery_level'] > combinations['battery_total_required']]
    if valid_combinations.empty:
        return

    best_assignments = greedy_two_pass_assignment(valid_combinations)
    best_assignments['eta_to_pickup'] = current_time + pd.to_timedelta(best_assignments['est_time_to_pickup'], unit='s')
    best_assignments['eta_to_dropoff'] = best_assignments['eta_to_pickup'] + pd.to_timedelta(
        best_assignments['trip_time'], unit='s')

    car_id_to_index = dict(zip(fleet['car_id'], fleet.index))
    assign_indices = best_assignments['car_id'].map(car_id_to_index)

    fleet_update_df = pd.DataFrame({
        'index': assign_indices,
        'current_trip_id': best_assignments['trip_id'].values,
        'eta_to_pickup': best_assignments['eta_to_pickup'].values,
        'eta_to_dropoff': best_assignments['eta_to_dropoff'].values,
        'battery_used': best_assignments['battery_total_required'].values,
    })

    fleet.loc[fleet_update_df['index'], 'status'] = 'en_route_to_client'
    fleet.loc[fleet_update_df['index'], 'current_trip_id'] = fleet_update_df['current_trip_id'].values
    fleet.loc[fleet_update_df['index'], 'eta_to_pickup'] = fleet_update_df['eta_to_pickup'].values
    fleet.loc[fleet_update_df['index'], 'eta_to_dropoff'] = fleet_update_df['eta_to_dropoff'].values
    fleet.loc[fleet_update_df['index'], 'battery_level'] -= fleet_update_df['battery_used'].values

    trips_data_day.loc[trips_data_day['trip_id'].isin(best_assignments['trip_id']), 'assigned'] = True

    trip_id_to_car = dict(zip(best_assignments['trip_id'], best_assignments['car_id']))
    trips_data_day.loc[trips_data_day['trip_id'].isin(trip_id_to_car.keys()), 'completed_by'] = \
        trips_data_day['trip_id'].map(trip_id_to_car)


def assign_to_charging(car_idx):
    """Assign car to the nearest available charging station."""
    available_stations = charging_stations.copy()

    car_location = fleet.iloc[car_idx]['location_id']

    available_stations = available_stations.merge(
        mean_distance_time[mean_distance_time['PULocationID'] == car_location],
        left_on='location_id',
        right_on='DOLocationID').rename(columns={'trip_time': 'trip_time_to_charger',
                                                 'trip_miles': 'trip_miles_to_charger'})


    available_stations["estimated_wait_time"] = available_stations['estimated_time_to_charge'].apply(lambda x: sum(x)) + \
                                                available_stations['waiting_to_be_charged'].apply(
                                                    lambda x: len(x)) * charging_time
    available_stations["available_slot"] = available_stations["occupied_slot"].apply(lambda x: any(x == 0 for x in x))
    available_stations.loc[available_stations["available_slot"] == True, "estimated_wait_time"] = 0

    available_stations["estimated_wait_time_to_charge"] = available_stations["trip_time_to_charger"] / 60 + \
                                                          available_stations["estimated_wait_time"]
    best_station = available_stations.nsmallest(1, "estimated_wait_time_to_charge").iloc[0]
    best_station_idx = charging_stations[charging_stations['station_id'] == best_station['station_id']].index[0]
    #TODO: подправить логику с зарядкой - сделать статус en_route_to_charge и учесть движение до станции
    # + подбирать на двух параметрах - estimated time to charge + trip_time
    try:
        empty_slot = best_station['occupied_slot'].index(0)
        fleet.loc[car_idx, "status"] = "charging"
        fleet.loc[car_idx, "charging_station_id"] = best_station['station_id']

        charging_stations.loc[best_station_idx, 'occupied_slot'][empty_slot] = fleet.iloc[car_idx]['car_id']
        charging_stations.loc[best_station_idx, \
            'estimated_time_to_charge'][empty_slot] = (distance_per_charge - fleet.iloc[car_idx]['battery_level']) / \
                                                      speed_of_charging

    except:
        charging_stations.loc[best_station_idx, 'waiting_to_be_charged'].append(fleet.iloc[car_idx]['car_id'])
        fleet.loc[car_idx, "status"] = "waiting_to_charge"


def update_car_status(trips_data, current_time):
    if (fleet["status"] == "en_route_to_client").any():
        mask_to_client = (fleet["status"] == "en_route_to_client") & (fleet["eta_to_pickup"] <= current_time)
        fleet.loc[mask_to_client, "status"] = "en_route_with_client"
        trip_ids = fleet.loc[mask_to_client, "current_trip_id"]
        trips_data.loc[trips_data['trip_id'].isin(trip_ids), 'actual_eta_pickup'] = current_time

    mask_with_client = (fleet["status"] == "en_route_with_client") & (pd.to_datetime(fleet["eta_to_dropoff"]) <= current_time)
    if mask_with_client.any():
        completed_fleet = fleet.loc[mask_with_client]
        fleet.loc[mask_with_client, "status"] = "idle"
        fleet.loc[mask_with_client, "current_trip_id"] = None
        trip_info = trips_data.set_index("trip_id").loc[completed_fleet["current_trip_id"]][["DOLocationID"]]
        fleet.loc[mask_with_client, "location_id"] = trip_info.values
        trips_data.loc[trips_data['trip_id'].isin(completed_fleet["current_trip_id"]), 'complete_time'] = current_time

    charging_mask = fleet["status"] == "charging"
    for idx in fleet[charging_mask].index:
        fleet.loc[idx, "battery_level"] += speed_of_charging
        car = fleet.loc[idx]
        station_id = car["charging_station_id"]
        station_idx = charging_stations[charging_stations['station_id'] == station_id].index[0]
        station = charging_stations.loc[station_idx]
        charging_slot = station['occupied_slot'].index(car['car_id'])
        est_charge_time = max(0, (distance_per_charge - car["battery_level"]) / speed_of_charging)
        charging_stations.at[station_idx, 'estimated_time_to_charge'][charging_slot] = est_charge_time

        if car["battery_level"] >= distance_per_charge:
            fleet.at[idx, "status"] = "idle"
            charging_stations.at[station_idx, 'occupied_slot'][charging_slot] = 0
            charging_stations.at[station_idx, 'estimated_time_to_charge'][charging_slot] = 0

            queue = charging_stations.at[station_idx, "waiting_to_be_charged"]
            if queue:
                next_car_id = queue.pop(0)
                next_car_idx = fleet[fleet['car_id'] == next_car_id].index[0]
                fleet.at[next_car_idx, "status"] = "charging"
                charging_stations.at[station_idx, 'occupied_slot'][charging_slot] = next_car_id
                charging_stations.at[station_idx, 'estimated_time_to_charge'][charging_slot] = (
                    distance_per_charge - fleet.at[next_car_idx, 'battery_level']) / speed_of_charging

    low_battery_mask = (fleet["battery_level"] <= 20) & (fleet["status"] == "idle")
    for idx in fleet[low_battery_mask].index:
        assign_to_charging(idx)


def run_daily_simulation(trips_data_day, start_time, fleet, charging_stations, minutes_in_day=1441):
    on_line_cars = pd.DataFrame()

    for t in range(minutes_in_day):
        current_time = start_time + pd.to_timedelta(t, unit="m")

        current_trips = trips_data_day[
            (trips_data_day["assigned"] == False) &
            ((current_time - trips_data_day['request_datetime']).dt.total_seconds() < max_wait_time) &
            ((current_time - trips_data_day['request_datetime']).dt.total_seconds() >= 0)
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

    return completed_trips, current_trips, lost_trips, on_line_cars, fleet, charging_stations


fleet_sizes = [100, 500, 1000, 2000, 3500]
charging_stations_counts = [20, 100, 200, 400, 700]

for i in range(len(fleet_sizes)):

    fleet_size = fleet_sizes[i]
    charging_stations_count = charging_stations_counts[i]
    charging_stations = pd.DataFrame({
        "location_id": random.choices(unique_locations, k=charging_stations_count),
        "occupied_slot": [[0 for _ in range(charging_station_capacity)] for _ in range(charging_stations_count)],
        "waiting_to_be_charged": [[] for _ in range(charging_stations_count)],
        "estimated_time_to_charge": [[0 for _ in range(charging_station_capacity)] for _ in
                                     range(charging_stations_count)],
        "station_id": [str(uuid.uuid4()) for _ in range(charging_stations_count)]

    })

    fleet = pd.DataFrame({
        "location_id": random.choices(unique_locations, k=fleet_size),
        "battery_level": [distance_per_charge] * fleet_size,  # Full battery
        "status": ["idle"] * fleet_size,  # idle, en_route_to_client, en_route_with_client, charging, waiting_to_charge
        "current_trip_id": [None] * fleet_size,
        "car_id": [str(uuid.uuid4()) for _ in range(fleet_size)],
        "eta_to_dropoff": [None] * fleet_size
    })

    current_trips = pd.DataFrame()

    folder_path = '../dataset/daily_trips'
    file_paths = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    )

    for file_path in file_paths:
        print(f"Running simulation for day {file_path}")
        start_time = datetime.strptime(file_path.split('_')[-1].split('.')[0], '%Y-%m-%d')

        day_trips = pd.concat([current_trips, pd.read_parquet(file_path)], ignore_index=True)

        completed_trips, current_trips, lost_trips, online_cars, fleet, charging_stations = run_daily_simulation(
            day_trips,
            start_time,
            fleet,
            charging_stations
        )

        final_completed_trips = pd.concat([completed_trips, lost_trips], axis=0)

        file_path = Path(f"../results/saev_completed_trips_day_{fleet_size}.parquet")
        if file_path.exists():
            final_completed_trips.to_parquet(f"../results/saev_completed_trips_day_{fleet_size}.parquet",
                                             engine='fastparquet', append=True, index=False)
            online_cars.to_parquet(f"../results/saev_online_cars_day_{fleet_size}.parquet", engine='fastparquet',
                                   append=True, index=False)
        else:
            final_completed_trips.to_parquet(f"../results/saev_completed_trips_day_{fleet_size}.parquet",
                                             engine='fastparquet', index=False)
            online_cars.to_parquet(f"../results/saev_online_cars_day_{fleet_size}.parquet", engine='fastparquet',
                                   index=False)

        fleet.to_csv(f"../setup/saev_fleet_final_{fleet_size}.csv", index=False)
        charging_stations.to_csv(f"../setup/saev_stations_final_{fleet_size}.csv", index=False)
