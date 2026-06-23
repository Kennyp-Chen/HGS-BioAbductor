import pandas as pd
name_tb="OS_60"
t = 60
df = pd.read_csv("TCGA/standardFileFoldTCGA/ClinicalDataFrame-Cut15Years.csv",index_col = 0)
fn_dt = "TCGA/standardFileFoldTCGA/test_dt.csv"
interval = (df["OS"].max()-df["OS"].min()+1) / t
print(interval)
# interPat_list = []
for i in range(t):
    top     = ((i+1)*interval+df["OS"].min())
    bottom  = (i * interval + df["OS"].min())
    # interPat_list.append((( df["OS"] >= bottom )==(   df["OS"] < top )).sum())
    df.loc[( df["OS"] >= bottom )==( df["OS"] < top ),name_tb] = i+1
df.to_csv(fn_dt,index=0)