import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# pandas でデータを読み込む。
df = pd.read_csv("ml-latest-small/ratings.csv")

# 疎行列(ほとんどの要素が0の行列)を作成する。ユーザ/アイテムPivot
df_rating = df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
# print(df_rating.head())

# コサイン類似度を求める
user_sim = cosine_similarity(df_rating)

# DataFrameにする (行・列ともに df_rating.index = userId指定)
user_sim_df = pd.DataFrame(user_sim, index=df_rating.index, columns=df_rating.index)

# print(user_sim_df.head())

# 対象者に近い人を降順でソート
print(user_sim_df[user_sim_df.index == 2].sort_values(by=2, axis=1, ascending=False))