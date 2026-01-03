import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

GM = 3.986e14 # Standard gravitational parameter for Earth (Gravitational constant * Mass)
EARTH_RADIUS = 6_378_137 # m

df = pd.read_csv("./data/25544_2020-2025.csv")

data = df[["MEAN_MOTION", "EPOCH", "INCLINATION"]].copy()

orbital_period = 86400 / data["MEAN_MOTION"]

data["ORBITAL_ALTITUDE"] = (
    ((GM * (orbital_period ** 2))/(4*np.pi ** 2)) ** (1/3)
      - EARTH_RADIUS 
    ) / 1000

# Overview
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

#Altitude data on y-axis
alt = data["ORBITAL_ALTITUDE"].to_list()

# Correlation coeficcient
coeff = np.polyfit(times, alt, 1)
function = np.poly1d(coeff)

# Plotting
plt.plot(epochs, alt, label="Orbital Altitude")
plt.plot(epochs, function(times), "--k")
plt.show()