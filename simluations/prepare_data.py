import os
import pandas as pd
import uuid

os.makedirs('../dataset/daily_trips', exist_ok=True)

trips_table = pd.read_parquet('../dataset/fhvhv_tripdata_2024-01 (1).parquet')
trips_table = trips_table[trips_table['request_datetime'] >= pd.to_datetime('2024-01-01')]


trips_table = trips_table[trips_table['shared_request_flag'] == 'N']
# trips_table = trips_table[trips_table['wav_request_flag'] == 'N']
# trips_table = trips_table[trips_table['access_a_ride_flag'] == 'N']

trips_table = trips_table[trips_table['request_datetime'] < trips_table['pickup_datetime']]

trips_table = trips_table[(trips_table['trip_time'] >= trips_table['trip_time'].quantile(0.05)) \
                          & (trips_table['trip_time'] <= trips_table['trip_time'].quantile(0.95)) \
                          & (trips_table['trip_miles'] >= trips_table['trip_miles'].quantile(0.05)) \
                          & (trips_table['trip_miles'] <= trips_table['trip_miles'].quantile(0.95)) \
                          & (trips_table['base_passenger_fare'] <= trips_table['base_passenger_fare'].quantile(0.95)) \
                          & (trips_table['base_passenger_fare'] >= trips_table['base_passenger_fare'].quantile(0.05)) \
                          ]
trips_table['trip_id'] = [str(uuid.uuid4()) for _ in range(len(trips_table))]
trips_table['completed_by'] = None

trips_table["assigned"] = False
trips_table['complete_time'] = pd.to_datetime('1999-01-01')


days = sorted(trips_table['request_datetime'].dt.date.unique())

mean_distance_time = trips_table.groupby(['PULocationID', 'DOLocationID']).agg({
    'trip_time': 'mean',
    'trip_miles': 'mean'
}).reset_index()
inverse_trips = mean_distance_time.rename(columns={
    'PULocationID': 'DOLocationID',
    'DOLocationID': 'PULocationID'
})
inverse_trips = inverse_trips[~inverse_trips.set_index(['PULocationID', 'DOLocationID']).index.isin(
    mean_distance_time.set_index(['PULocationID', 'DOLocationID']).index)]
mean_distance_time = pd.concat([mean_distance_time, inverse_trips], ignore_index=True)

# Extract unique location IDs
location_ids = pd.concat([trips_table['PULocationID'], trips_table['DOLocationID']]).unique()

all_combinations = pd.DataFrame([(pul, dol) for pul in location_ids for dol in location_ids],
                                columns=['PULocationID', 'DOLocationID'])

mean_distance_time_complete = all_combinations.merge(
    mean_distance_time,
    on=['PULocationID', 'DOLocationID'],
    how='left'
)

mean_trip_time = mean_distance_time['trip_time'].mean()
mean_trip_miles = mean_distance_time['trip_miles'].mean()

mean_distance_time_complete['trip_time'].fillna(mean_trip_time, inplace=True)
mean_distance_time_complete['trip_miles'].fillna(mean_trip_miles, inplace=True)

mean_distance_time = mean_distance_time_complete

max_location_id = trips_table["PULocationID"].max()

mean_distance_time.to_csv('../dataset/mean_distance_time.csv', index=False)

trips_table.to_parquet(f"../dataset/prepared_trips.parquet", index=False)
print(len(trips_table))

for date in days:
    daily_df = trips_table[trips_table['request_datetime'].dt.date == date].sample(100000)
    path = f"../dataset/daily_trips/trips_{date}.parquet"
    daily_df.to_parquet(path, index=False)