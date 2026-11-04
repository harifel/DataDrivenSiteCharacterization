
# %% [00] Import packages
# --------------------------------------------------------------------------
import pandas as pd
from SBT import get_SBT, robertson2010_plot

# %% [00] Read file
# --------------------------------------------------------------------------
path = r"C:\Users\niego\Nextcloud\Shared\Master's thesis_Ali Yaz\05_Workpackage4\data\site_Perniö\github"   # Folder
file = r"02_Pernio_CPT.csv"                                        # Filename
CPT_df = pd.read_csv(rf"{path}\{file}", dtype={"Dataset": str, "ID": str})

# %% [01] Get SBT
# --------------------------------------------------------------------------
depth = CPT_df["Depth (m)"]
qc = CPT_df["qc (MPa)"]
Rf = CPT_df["Rf (%)"]
u2 = CPT_df["u2 (kPa)"]

# Convert kPa to MPa (if needed)
u2 = u2 / 1000                             

# Net cone resistance, qt (MPa)
qt = qc + (1-0.80) * u2                                 # Assuming a0 = 0.80


SBTs = get_SBT(Rf, qt)
CPT_df["SBT"] = SBTs
# %% [02] Plot Robertson Chart
# --------------------------------------------------------------------------

CPT_df_plot = pd.DataFrame({
    'depth': depth,   
    'qt': qt,
    'Rf': Rf})
    
robertson2010_plot(CPT_df_plot)

# %% [03] Export
# --------------------------------------------------------------------------
export_filename = rf"{file.split('.')[0]}_test.csv"
CPT_df.to_csv(rf"{path}\{export_filename}", index=False)

# %% [END]
