import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

GM = 3.986e14 # Standard gravitational parameter for Earth (Gravitational constant * Mass)
EARTH_RADIUS = 6_378_137 # Metres

df = pd.read_csv("./data/25544_2020-2025.csv")


data = df[["MEAN_MOTION", "EPOCH", "INCLINATION"]].copy()


orbital_altitudes = []


for i in data.index:
    mean_motion = data.loc[i]["MEAN_MOTION"] # rev/day
    
    orbital_period = (24 * 60 * 60)/mean_motion
    print(f"orbital_period = {orbital_period}")

    orbital_altitude = ((GM * (orbital_period ** 2))/(4*np.pi ** 2)) ** (1/3) - EARTH_RADIUS # Metres 
    orbital_altitude = orbital_altitude / 1000 # KM
    print(f"orbital_altitude = {orbital_altitude}")

    orbital_altitudes.append(orbital_altitude)

# Overview
data["ORBITAL_ALTITUDE"] = orbital_altitudes


max_altitude, min_altitude = max(orbital_altitudes), min(orbital_altitudes)
altitude_variation = max_altitude - min_altitude
print(f"""
OVERVIEW:
    Maximum Altitude: {max_altitude}
    Minimum Altitude: {min_altitude}
    Altitude Variation: {altitude_variation}
""")
#Rolling average ("Smoothing" the data)
# data["SMOOTH_ALTITUDE"] = data["ORBITAL_ALTITUDE"].rolling(5000).mean()


#Linear Regression


#Epochs on x-axis
epochs = pd.to_datetime(data["EPOCH"])

time_deltas = (epochs - epochs.iloc[0])

times = time_deltas.dt.total_seconds().to_numpy()

#Altitude data on y-axis
alt = data["ORBITAL_ALTITUDE"].to_list()

# Correlation coeficcient
coeff = np.polyfit(times, alt, 1)
function = np.poly1d(coeff)

# Plotting
# data.plot(x="EPOCH", y="ORBITAL_ALTITUDE")
# data.plot(x="EPOCH", y="SMOOTH_ALTITUDE")
plt.plot(epochs, alt, label="Orbital Altitude")
plt.plot(epochs, function(times), "--k")
plt.show()


