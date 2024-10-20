# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/mnt/data/Uber-Jan-Feb-FOIL.csv'
uber_df = pd.read_csv(file_path) 

uber_df['Date/Time'] = pd.to_datetime(uber_df['Date/Time'], format='%m/%d/%Y %H:%M:%S')

uber_df['hour'] = uber_df['Date/Time'].dt.hour
uber_df['day'] = uber_df['Date/Time'].dt.day
uber_df['month'] = uber_df['Date/Time'].dt.month
uber_df['day_of_week'] = uber_df['Date/Time'].dt.dayofweek

trips_per_hour = uber_df.groupby('hour').size()
plt.figure(figsize=(10, 6))
trips_per_hour.plot(kind='bar')
plt.title('Number of Trips per Hour')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()

trips_per_day = uber_df.groupby('day').size()
plt.figure(figsize=(10, 6))
trips_per_day.plot(kind='bar')
plt.title('Number of Trips per Day')
plt.xlabel('Day')
plt.ylabel('Number of Trips')
plt.show()

trips_per_month = uber_df.groupby('month').size()
plt.figure(figsize=(10, 6))
trips_per_month.plot(kind='bar')
plt.title('Number of Trips per Month')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.show()

trips_per_weekday = uber_df.groupby('day_of_week').size()
plt.figure(figsize=(10, 6))
trips_per_weekday.plot(kind='bar', color='green')
plt.title('Number of Trips by Day of the Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Number of Trips')
plt.show()

heatmap_data = uber_df.groupby(['day_of_week', 'hour']).size().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu")
plt.title('Heatmap of Trips by Hour and Day of the Week')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(uber_df['Lat'], uber_df['Lon'], alpha=0.5, s=1)
plt.title('Scatter Plot of Uber Trips (Latitude vs Longitude)')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(uber_df['Lat'], uber_df['Lon'], cmap="Reds", shade=True, bw_adjust=.5)
plt.title('Heatmap of Popular Pickup Locations')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

base_distribution = uber_df['Base'].value_counts()
plt.figure(figsize=(10, 6))
base_distribution.plot(kind='bar', color='purple')
plt.title('Distribution of Trips by Base')
plt.xlabel('Base')
plt.ylabel('Number of Trips')
plt.show()

mean_lat_lon = uber_df.groupby('hour')[['Lat', 'Lon']].mean()
plt.figure(figsize=(10, 6))
plt.plot(mean_lat_lon['Lat'], label='Latitude', color='orange')
plt.plot(mean_lat_lon['Lon'], label='Longitude', color='blue')
plt.title('Mean Latitude and Longitude by Hour')
plt.xlabel('Hour')
plt.ylabel('Mean Latitude/Longitude')
plt.legend()
plt.show()

most_freq_pickup_hour = uber_df['hour'].value_counts().idxmax()
print(f'Most frequent pickup hour: {most_freq_pickup_hour}')

most_freq_day = uber_df['day_of_week'].value_counts().idxmax()
print(f'Most frequent day of the week: {most_freq_day}')

plt.figure(figsize=(10, 6))
sns.heatmap(uber_df[['Lat', 'Lon', 'hour', 'day', 'month']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

uber_df['is_weekend'] = uber_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
weekend_weekday_trips = uber_df.groupby('is_weekend').size()
plt.figure(figsize=(10, 6))
weekend_weekday_trips.plot(kind='bar', color='brown')
plt.title('Trips on Weekends vs Weekdays')
plt.xlabel('0 = Weekday, 1 = Weekend')
plt.ylabel('Number of Trips')
plt.show()

weekend_data = uber_df[uber_df['is_weekend'] == 1]
hourly_distribution_weekends = weekend_data.groupby('hour').size()
plt.figure(figsize=(10, 6))
hourly_distribution_weekends.plot(kind='bar', color='teal')
plt.title('Hourly Distribution of Trips on Weekends')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()

weekday_data = uber_df[uber_df['is_weekend'] == 0]
hourly_distribution_weekdays = weekday_data.groupby('hour').size()
plt.figure(figsize=(10, 6))
hourly_distribution_weekdays.plot(kind='bar', color='violet')
plt.title('Hourly Distribution of Trips on Weekdays')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()

top_5_hours = uber_df['hour'].value_counts().nlargest(5)
print(f'Top 5 busiest hours: \n{top_5_hours}')

plt.figure(figsize=(10, 6))
sns.countplot(x='day', hue='is_weekend', data=uber_df, palette='muted')
plt.title('Day-wise Count of Trips (Weekday vs Weekend)')
plt.xlabel('Day')
plt.ylabel('Number of Trips')
plt.show()

uber_df['distance'] = np.sqrt((uber_df['Lat'] - uber_df['Lat'].mean())**2 + (uber_df['Lon'] - uber_df['Lon'].mean())**2)
plt.figure(figsize=(10, 6))
sns.histplot(uber_df['distance'], bins=50, kde=True)
plt.title('Distribution of Trip Distances')
plt.xlabel('Distance (Approximation)')
plt.ylabel('Frequency')
plt.show()


uber_df.head()
