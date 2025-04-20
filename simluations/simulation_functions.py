from datetime import timedelta, datetime
import uuid
import os
from pathlib import Path
import pyarrow.parquet as pq

import pandas as pd

import random

def greedy_two_pass_assignment(valid_combinations):
    valid_combinations = valid_combinations.sort_values('est_time_to_pickup').copy()

    first_pass = valid_combinations.drop_duplicates(subset='trip_id', keep='first')
    first_pass = first_pass.drop_duplicates(subset='car_id', keep='first')

    assigned_trips = set(first_pass['trip_id'])
    assigned_cars = set(first_pass['car_id'])

    remaining_combos = valid_combinations[
        ~valid_combinations['trip_id'].isin(assigned_trips) &
        ~valid_combinations['car_id'].isin(assigned_cars)
        ].copy()

    second_pass = []

    if not remaining_combos.empty:
        remaining_combos = remaining_combos.sort_values('est_time_to_pickup').copy()

        while not remaining_combos.empty:
            mask = (~remaining_combos['trip_id'].isin(assigned_trips)) & (
                ~remaining_combos['car_id'].isin(assigned_cars))
            if not mask.any():
                break

            valid_rows = remaining_combos[mask]
            if valid_rows.empty:
                break

            best_row = valid_rows.iloc[0]
            second_pass.append(best_row)

            assigned_trips.add(best_row['trip_id'])
            assigned_cars.add(best_row['car_id'])

            remaining_combos = remaining_combos[
                (remaining_combos['trip_id'] != best_row['trip_id']) &
                (remaining_combos['car_id'] != best_row['car_id'])
                ]

    second_pass_df = pd.DataFrame(second_pass)
    best_assignments = pd.concat([first_pass, second_pass_df], ignore_index=True)
    return best_assignments


def calculate_on_line_cars(fleet, on_line_cars, current_time):
    online = fleet[fleet["status"].isin(["idle", "en_route_to_client", "en_route_with_client"])]
    return pd.concat([on_line_cars, online[['car_id']].assign(timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S"))])