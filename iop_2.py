import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from icecream import ic


def set_labels(ax: Axes, subtitle: str, legend_loc: str = "best"):
    ax.set_xlabel("Angle [degrees]")
    ax.set_ylabel("Signal [dB]")
    ax.set_title(subtitle, size=18)
    ax.legend(loc=legend_loc)


antenna_data = pd.read_csv("measured_data_tamara_clelford.csv")
antenna_data.info()

sns.set_style("ticks")
sns.set_palette("muted")
sns.set_context("notebook")  # notebook, paper, poster

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharey=True)

ax1 = ax[0]
ax2 = ax[1]

# plot measured data from csv
sns.lineplot(antenna_data, x="angle_deg", y="sum_m_db", label="Sum", ax=ax1)
sns.lineplot(
    antenna_data, x="angle_deg", y="diff_m_db", label="Diff", ax=ax1, linestyle="--"
)
set_labels(ax1, "Measured data")


# create simulated data
diameter = 0.15  # m
channel_offset = 0.05  # rad
wavelength = 3e8 / 15e9  # m

angle_rad = np.radians(np.arange(-30, 30, 1))  # range start,stop,step (inclusive)

u_left = ((np.pi * diameter) / wavelength) * (angle_rad - channel_offset)
left = np.sin(u_left) / u_left

u_right = ((np.pi * diameter) / wavelength) * (angle_rad + channel_offset)
right = np.sin(u_right) / u_right

antenna_sim = pd.DataFrame([angle_rad, left, right]).T
antenna_sim.columns = ["angle_rad", "left", "right"]


# create func to apply to df columns
def volts_to_db(volts: float):
    dB = 10 * (np.log10(volts**2))
    if dB == np.NINF:  # sns joins lines over nan, approximate -inf to -50dB
        return -50
    return dB


# antenna_sim["left_db"] = antenna_sim["left"].apply(volts_to_db)
# antenna_sim["right_db"] = antenna_sim["right"].apply(volts_to_db)

# convert back to degrees and cast as int for merge compatibility
antenna_sim["angle_deg"] = (
    antenna_sim["angle_rad"].apply(lambda x: (x * 180) / np.pi).astype(int)
)

# add voltage then convert to dB (or use appropriate formula of adding in log space)
# use apply(func) to return modified col
antenna_sim["sum_s_db"] = (antenna_sim["left"] + antenna_sim["right"]).apply(
    volts_to_db
)
antenna_sim["diff_s_db"] = (antenna_sim["left"] - antenna_sim["right"]).apply(
    volts_to_db
)

sns.lineplot(antenna_sim, x="angle_deg", y="sum_s_db", label="Sum", ax=ax2)
sns.lineplot(
    antenna_sim, x="angle_deg", y="diff_s_db", label="Diff", ax=ax2, linestyle="--"
)


set_labels(ax2, "Simulated data")

fig.subplots_adjust(hspace=0.4)
fig.suptitle("Antenna signal vs angle for real and simulated data", size=24)

plt.savefig("antennas_comparison.png", dpi=300)
plt.show()

# don't incluce angle_rad
antenna_sim_reduced = antenna_sim[
    ["angle_deg", "sum_s_db", "diff_s_db"]
]  # select columns to inner join to new df

save_df = antenna_data.merge(
    antenna_sim_reduced, on="angle_deg", how="inner"
)  # merge into dataframe containing both datasets

save_df.to_csv("antennas_real_sim.csv", index=False)
