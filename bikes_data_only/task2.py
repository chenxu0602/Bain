
import numpy as np
import pandas as pd

df_station = pd.read_csv("data/station_data.csv")
df_trip = pd.read_csv("data/trip_data.csv", parse_dates=["Start Date", "End Date"], infer_datetime_format=True)
df_weather = pd.read_csv("data/weather_data.csv", parse_dates=["Date"], infer_datetime_format=True)

# Moved stations
moves = {
    23: 85,
    25: 86,
    49: 87,
    69: 88,
    72: 90,
    89: 90,
}

for o, n in moves.items():
    df_trip.loc[df_trip["Start Station"] == o, "Start Station"] = n
    df_trip.loc[df_trip["End Station"] == o, "End Station"] = n

""" Bin one day into 24 hours and count the the difference of starts and ends for each station in each hour divided by 365. """
""" Using the mean could be the benchmark """

""" Get the date and hour part the time timestamp """
df_trip["Date"] = df_trip["Start Date"].dt.date
df_trip["Start Hour"] = df_trip["Start Date"].dt.hour
df_trip["End Hour"] = df_trip["End Date"].dt.hour

""" Weekends usage seems to drop a lot so remove weekends data """
df_trip["Weekday"] = df_trip["Start Date"].dt.weekday
df_trip = df_trip.loc[df_trip.Weekday < 5]

""" Count by station and hour the number of start and returns """
df_start_hour_count = df_trip.groupby(["Start Station", "Start Hour"])["Start Date"].agg("count")
df_end_hour_count = df_trip.groupby(["End Station", "End Hour"])["End Date"].agg("count")

df_start_hour_count = pd.DataFrame({"start": df_start_hour_count})
df_start_hour_count.reset_index(inplace=True)
df_start_hour_count.columns = ["Id", "Hour", "Start Count"]

df_end_hour_count = pd.DataFrame({"end": df_end_hour_count})
df_end_hour_count.reset_index(inplace=True)
df_end_hour_count.columns = ["Id", "Hour", "End Count"]

""" The net rate is the simply the difference between the start and the end """
df_hour_count = pd.merge(df_start_hour_count, df_end_hour_count, on=["Id", "Hour"])
df_hour_count["Net Rate"] = (df_hour_count["End Count"] - df_hour_count["Start Count"]) / (52 * 5.)

def net_rate(station, hour):
    return df_hour_count.loc[(df_hour_count["Id"] == station) & (df_hour_count["Hour"] == hour), "Net Rate"].values[0] 


#Calculate RMSE
df_start_date_hour_count = df_trip.groupby(["Start Station", "Date", "Start Hour"])["Start Date"].agg("count")
df_end_date_hour_count = df_trip.groupby(["End Station", "Date", "End Hour"])["End Date"].agg("count")

df_start_date_hour_count = pd.DataFrame({"start": df_start_date_hour_count})
df_start_date_hour_count.reset_index(inplace=True)
df_start_date_hour_count.columns = ["Id", "Date", "Hour", "Start Count"]

df_end_date_hour_count = pd.DataFrame({"end": df_end_date_hour_count})
df_end_date_hour_count.reset_index(inplace=True)
df_end_date_hour_count.columns = ["Id", "Date", "Hour", "End Count"]

df_date_hour_count = pd.merge(df_start_date_hour_count, df_end_date_hour_count, on=["Id", "Date", "Hour"])
df_date_hour_count["Net Rate"] = df_date_hour_count["End Count"] - df_date_hour_count["Start Count"]

df_date_hour_count2 = pd.merge(df_date_hour_count, df_hour_count, on=["Id", "Hour"], how="left")
RMSE = np.sqrt(np.power(df_date_hour_count2["Net Rate_x"] - df_date_hour_count2["Net Rate_y"], 2).sum() / len(df_date_hour_count2))

print("RMSE: ", RMSE)


""" The RMSE is too big compared to the average net rate so the result is meaningless """

"""Excluding holidays there are roughly 70 * 252 * 24 = 423360 station hours in one year. """
""" However, in dataframe df_date_hour_count I found there are only 9356 station hours that have abs(Net Rate) >= 5 """
""" Most of the time the net rate is close to 0 and negligible. The problem should be to forecasting the rare events that """
""" net rate becomes large (either positive or negative) in the next hour. """
""" Actually the two peaks of net rate centers around 8AM and 4-5PM. We should use this features to build a model."""

