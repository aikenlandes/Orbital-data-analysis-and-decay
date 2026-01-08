import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

GM = 3.986e14 # Standard gravitational parameter for Earth (Gravitational constant * Mass)
EARTH_RADIUS = 6_378_137 # m

df1 = pd.read_csv("./data/2007-2015.csv")
df2 = pd.read_csv("./data/2015-2019.csv")
df3 = pd.read_csv("./data/2020-2025.csv")

df = pd.concat([df1, df2, df3])

data = df[["MEAN_MOTION", "EPOCH", "INCLINATION"]].copy()

orbital_period = 86400 / data["MEAN_MOTION"]

data["ORBITAL_ALTITUDE"] = (
    ((GM * (orbital_period ** 2))/(4*np.pi ** 2)) ** (1/3)
      - EARTH_RADIUS 
    ) / 1000

# Overview
data["ALTITUDE_SMOOTH"] = data["ORBITAL_ALTITUDE"].rolling(window=250).mean()
max_altitude, min_altitude = data["ORBITAL_ALTITUDE"].max(), data["ORBITAL_ALTITUDE"].min()
altitude_variation = max_altitude - min_altitude
print(f"""
OVERVIEW:
    Maximum Altitude: {max_altitude}
    Minimum Altitude: {min_altitude}
    Altitude Variation: {altitude_variation}
""")

#Linear Regression
epochs = pd.to_datetime(data["EPOCH"])#Epochs on x-axis
time_delta = (epochs - epochs.iloc[0])#Time since start for each record
times = time_delta.dt.total_seconds().to_numpy()

# Clean up NAN values
mask = data["ALTITUDE_SMOOTH"].notna()

times_clean = times[mask.to_numpy()]
alt_clean = data.loc[mask, "ALTITUDE_SMOOTH"].to_numpy()

#Altitude data on y-axis
alt = data["ALTITUDE_SMOOTH"].to_list()
# Correlation coeficcient
coeff = np.polyfit(times_clean, alt_clean, 1)
decay_km_per_day = coeff[0] * 86400
print(f"Decay rate: {decay_km_per_day:.4f} km/day")
function = np.poly1d(coeff)

#Plotting
plt.plot(epochs, data["ORBITAL_ALTITUDE"], alpha=0.4, label="Raw Altitude")
plt.plot(epochs, data["ALTITUDE_SMOOTH"], label="Smoothed Altitude")
plt.plot(epochs[mask], function(times_clean), "--k", label="Decay Trend")

plt.xlabel("Epoch")
plt.ylabel("Altitude (km)")
plt.legend()
plt.show()
