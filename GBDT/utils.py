import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import copy

def scaling(X,X0,zscore_cols,position_cols):
	for col in zscore_cols:
		X[col]=(X[col]-X0[col].mean())/X0[col].std()

	for col in position_cols:
		X[col]=X[col]/X0[col].max()

	return X


def create_cluster(df_is, df_os, colname, k, max_iter=300, n_init=10, outname=None):
	if outname is None:
		outname="cluster_k_"+len(colname)

	df_is_is = copy.deepcopy(df_is)
	df_os_os = copy.deepcopy(df_os)
	
	for s in colname:
		df_is_is[s] = df_is[s].fillna(df_is[s].mean())
		df_os_os[s] = df_os[s].fillna(df_is[s].mean())

	km_cluster = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, \
		init='k-means++', verbose=100, n_jobs=-1)


	array_is = np.array(df_is_is[colname], dtype=float)
	array_os = np.array(df_os_os[colname], dtype=float)

	#print(np.isnan(array_is).any())

	km_cluster.fit(array_is)

	df_is[outname] = km_cluster.predict(array_is)
	df_os[outname] = km_cluster.predict(array_os)

	return df_is, df_os


def clean_df(df,obj = None):
	if obj is None:
		obj = df
	for s in df.columns:
		df[s] = df[s].fillna(obj[s].mean())
	return df
