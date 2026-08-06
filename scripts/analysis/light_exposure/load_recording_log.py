# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: eyewire2-functional-analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Recording log from QDSpy log and `smh` headers
#
# This notebook read and parses a stimulus log file (`QDSpy.ini`) and the ScanM header files (`*.smh`) and writes thie contained information in pandas DataFrames as well as `.csv` files for further processing.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
from datetime import datetime, time
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# %% [markdown]
# Set path to log file name and folder with `.smh` files

# %%
from eyewire2_functional_analysis.data_loader import DATA_2P

LOG_PATH = Path(DATA_2P) / "stimuli" / "20181011_182540.log"
SMH_PATH = Path(DATA_2P) / "smh"

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## Parse QDSpy log file
# Extract information from log file, resulting in a list of stimulus presentations (`stims`, list of dictionaries) will timing information and stimulus parameters

# %%
nLinesTotal = 0
nLinesData = 0
nLinesErr = 0
nErr = 0
isStimStarted = False
stims = []
nStims = 0

with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as fLog:
    for line in fLog:
        # Extract elements of each line
        nLinesTotal += 1
        sDateTime = line[:15]
        sInfoType = line[15:23].strip().upper()
        sMsg = line[23:len(line) -1].strip()

        # Convert time stamp into a datetime        
        dt = datetime.strptime(sDateTime, "%Y%m%d_%H%M%S")
        if nLinesTotal == 1:
            # First line; take as start time
            dt_log_start = dt
            dt_last_end = dt_log_start

        # Filter for relevant information
        if sInfoType not in ["DATA"]:
            # Ignore filed
            continue

        # Convert data line into dictionary 
        sMsg = sMsg.replace("'", "\"")
        sMsg = sMsg.replace("\\\\", "/")
        sMsg = sMsg.replace("(", "[")
        sMsg = sMsg.replace(")", "]")

        try:
            data = json.loads(sMsg)
        except json.JSONDecodeError as e:
            # JSON parsing failed
            print(f"ERROR: parsing line {nLinesTotal-1} failed:")
            print(f"'{sMsg}'")
            nLinesErr += 1
            data = None

        # Get stimulus start/stop pairs
        try:
            stimState = data["stimState"].upper()
        except KeyError:
            stimState = None        


        if stimState:
            # Data contains stimulus information
            if stimState == "STARTED":
                if isStimStarted:
                    print("ERROR: Two consecutive stimulus starts")
                    nErr += 1

                isStimStarted = True
                iLineLastStart = nLinesData
                dt_start = dt
                t_diff = (dt -dt_log_start).total_seconds()
                t_diff_last = (dt -dt_last_end).total_seconds()
                stimInfo = dict(
                    {"index": nStims, 
                     "stimFileName": Path(data["stimFileName"]).name,
                     "stimPath": str(Path(data["stimFileName"]).parent),
                     "stimMD5": data["stimMD5"],
                     "t_abs_s": t_diff,
                     "t_since_last_s": t_diff_last,
                     "t_start": dt.time()}

                )

            elif stimState in ["ABORTED", "FINISHED"]:
                if isStimStarted:
                    # Check if stimulus end belongs to stimulus start
                    fn = str(Path(stimInfo["stimPath"], stimInfo["stimFileName"])).replace("\\", "/")
                    if not data["stimFileName"] == fn:
                        print("ERROR: File paths for stimulus start and end differ")
                        nErr += 1

                    # Append stimulus list entry
                    dt_last_end = dt
                    t_diff = (dt -dt_start).total_seconds()
                    stimInfo.update(
                        {"aborted": stimState == "ABORTED",
                         "t_end": dt.time(),
                         "t_dur_s": t_diff}
                    )
                    stims.append(stimInfo)
                    nStims += 1
                    isStimStarted = False
                else:
                    print("ERROR: Stimulus end w/o start?")    
                    nErr += 1
        else:
            # Other information
            try:
                _ = data["nFrames"]
                isFrameInfo = True
            except KeyError:
                isFrameInfo = False

            if isFrameInfo:
                # Information about stimulus presentation statistics
                stims[nStims -1].update(
                    {"t_dur_s_calc": data["nFrames"] /data["avgFreq_Hz"],
                     "nDroppedFrames": data["nDroppedFrames"]}
                )
            else:            
                if nLinesData > iLineLastStart and nLinesData < iLineLastStart +3:
                    # Last start was only up to 2 lines before
                    stims[nStims -1].update(
                        {"params": data}
                    )
                else:
                    print("ERROR: Data w/o start??")    
                    nErr += 1

        '''
        print(f"line #{nLinesData}:")
        print(dt)
        print(data)
        '''
        nLinesData += 1

    print(f"{nLinesData} of {nLinesTotal} line(s) extracted.")
    print(f"{nLinesErr} line(s) failed parsing, {nErr} error(s) occurred post-processing.")    

# %% [markdown]
# Print stimulus presentation table

# %%
print("  #    start   t [s] gap [s] dur [s] nFr/frq abort stimulus name")
print("--- -------- ------- ------- ------- ------- ----- ")

for stim in stims:
    s = f"{stim["index"]:3d} {stim["t_start"]} {stim["t_abs_s"]:7.0f} "
    s += f"{stim["t_since_last_s"]:7.0f} {stim["t_dur_s"]:7.0f} {stim["t_dur_s_calc"]:7.0f} "  
    s += f"{"  y   " if stim["aborted"] else "      "} "
    s += f"`{Path(stim["stimFileName"]).name}`"

    print(s)

# %% [markdown]
# Example for information contained in one stimulus presentation entry

# %%
stims[1]

# %% [markdown]
# ... and as a pandas DataFrame, which is then written as a `.csv`file

# %%
df_stims = pd.DataFrame(stims)
df_stims.to_csv("stims.csv", sep=";", decimal=",")
df_stims


# %% [markdown]
# ## Parse ScanM `.smh` header files 
# ... to extract recording informations

# %%
def stamps2datetime(s_date :str, s_time :str) -> datetime:
    """ Convert date and time stamps from .SMH header files 
    """
    d = datetime.strptime(s_date, '%Y-%m-%d').date()
    h, m, s, *_ = s_time.split('-')    
    return datetime.combine(d, time(int(h), int(m), int(s)))

def sec_to_hms(sec):
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second


# %%
from eyewire2_functional_analysis.scanm.scanm_smp import SMP

nFiles = 0
smh_files = list(SMH_PATH.glob("*.smh"))
scmf = SMP()
smhs = []

# Loop through files and read data
for smh in smh_files:
    print(f"{nFiles:4d}: Reading {smh.name} ...")

    err_load_smh = scmf.loadSMH(smh, verbose=False)
    nFiles += 1

    dt = stamps2datetime(scmf._kvPairDict["DateStamp"][2], scmf._kvPairDict["TimeStamp"][2])
    info = dict(
        {"fName": smh.name,
         "date": dt.date(),
         "time": dt.time(),
         "xyz_coord_um": [
           scmf._kvPairDict["XCoord_um"][2], 
           scmf._kvPairDict["YCoord_um"][2], 
           scmf._kvPairDict["ZCoord_um"][2]]}
    )
    smhs.append(info)
    nFiles += 1

# %% [markdown]
# ... as pandas DataFrame, written into a `.csv` file

# %%
df_smhs = pd.DataFrame(smhs)
df_smhs.to_csv("smhs.csv", sep=";", decimal=",")
df_smhs

# %% [markdown]
# Plot recording field positions from `.smh` files

# %%
df = pd.DataFrame(smhs)

# Keep only rows with valid xyz_coord_um lists
df = df[df['xyz_coord_um'].notna()].copy()

# extract x, y (assumes [x,y,z] or similar)
df['x'] = df['xyz_coord_um'].apply(lambda v: float(v[0]) if v and len(v) >= 2 else np.nan)
df['y'] = df['xyz_coord_um'].apply(lambda v: float(v[1]) if v and len(v) >= 2 else np.nan)

df['t_seconds'] = df['time'].apply(time_to_seconds)

# drop invalid rows
df_plot = df.dropna(subset=['x', 'y', 't_seconds']).copy()

fig, ax = plt.subplots(figsize=(7,7))
sc = ax.scatter(df_plot['x'], df_plot['y'], c=df_plot['t_seconds'], cmap='viridis', s=50, edgecolor='k', lw=0.3)
cbar = fig.colorbar(sc, ax=ax)

# format colorbar ticks as HH:MM:SS
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: sec_to_hms(val)))
cbar.set_label('time (HH:MM:SS)')

ax.set_xlabel('X [um]')
ax.set_ylabel('Y [um]')
ax.set_title('SMH files: X vs Y (colored by time)')
ax.grid(True)

# enforce aspect ratio 1:1 and preserve it when resizing
ax.set_aspect(1.0)
ax.set_adjustable('box')

plt.savefig(os.path.join(FIG_DIR, 'smh_positions_by_time.pdf'))
plt.show()

# %%
