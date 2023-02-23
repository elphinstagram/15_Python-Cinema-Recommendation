# ユーザベースの協調フィルタリングで映画をリコメンドする
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# pandas でデータを読み込む。
df = pd.read_csv("ml-latest-small/ratings.csv")
# print(df.head())

# 疎行列(ほとんどの要素が0の行列)を作成する。ユーザ/アイテムPivot
df_rating = df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
print(df_rating.head())

# 最近傍探索で近い映画を検索する
neigh = NearestNeighbors(metric="cosine")

# 学習する
neigh.fit(df_rating)

# 特定の映画に距離が近い点を探索する
distnace, indices = neigh.kneighbors(df_rating[df_rating.index == 2])

# 2次元リスト
print(indices)

# 1次元に変換 (numpy利用)
print(indices.flatten())

# movieID で表示
for i in indices.flatten():
    print(df_rating.index[i])

# 近い人の評価点が高い映画を表示
movies_list = df[df["userId"] == 366]

# 点数の高い順に表示
print(movies_list.sort_values(by="rating", ascending=False))
