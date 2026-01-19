import pandas as pd

data = {
    "age": [25, 30, 45],
    "glucose": [85, 140, 160],
    "diabetes": [0, 1, 1]
}

df = pd.DataFrame(data)
print(df)
