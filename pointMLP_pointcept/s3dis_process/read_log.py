import pandas as pd

df = pd.read_csv("log.txt", sep="\t")
df.columns = [c.strip() for c in df.columns]  # 去掉列名空格
print(df)