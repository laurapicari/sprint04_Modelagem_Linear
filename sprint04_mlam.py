import os, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.dataset_download("itsrohithere/renewable-energy-production-dataset-2010-2020")
csv_file = glob.glob(os.path.join(path, "*.csv"))[0]

df = pd.read_csv(csv_file)
print(df.shape)
print(df.head())
df.info()
print(df.isna().sum())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)