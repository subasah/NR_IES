import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("da_hrl_lmps.csv")
df_wind = pd.read_csv("wind_gen.csv")

# print(df.head(10))
# print(df.head(10))
# Print the first value in the `fat` column

# print(df['total_lmp_da'].head(100))
print(df['total_lmp_da'].values.tolist())

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(df_wind['wind_generation_mw'].head(100))
print(df_wind['wind_generation_mw'].values.tolist())

