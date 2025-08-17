# -*- coding: utf-8 -*-
# @Date:   2024-08-05 22:54:07
# @Last Modified time: 2024-08-05 23:03:49
# @All Rights Reserved!

import statsmodels.tsa.seasonal as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 季节趋势分解，Seasonal-Trend decomposition
class STL(object):
	"""docstring for STL"""
	# period：每个周期的数量
	# periodic：周期，为True时，采用季节项均值
	def __init__(self, y, period, periodic = True):
		self.y = y
		stl = sm.STL(y, period = period)
		res = stl.fit()
		self.seasonal = res.seasonal # 季节项
		self.trend = res.trend # 趋势项
		self.residual = res.resid # 残差项
		if periodic:
			self.seasonal = self.seasonal_avg(self.seasonal, period)
			self.residual = self.y - self.seasonal - self.trend

		# 可视化
		# self.plot()

	def seasonal_avg(self, x, period):
		n = x.shape[0]
		# 周期数
		n_periods = int(n / period)
		# 数据不够一个周期时，直接返回
		if n_periods <= 1:
			return x
		use_cut = n_periods * period != n
		if use_cut:
			cut_len = period * n_periods
			mat = x[:cut_len]
		else:
			mat = x

		# 计算每一period对应位置的均值
		avg = np.mean(mat.reshape(n_periods, period), axis=0)
		# 将avg重复n_periods + 1次
		retval = np.tile(avg, n_periods + 1)
		retval = retval[:n]
		return retval

	def plot(self):
		x = range(len(self.y))
		figsz = (16, 10)
		fig = plt.figure(figsize = figsz)
		ax = fig.add_subplot(4, 1, 1)
		# plot Y
		ax.set_title("Observed")
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.plot(x, self.y)
		ax.set_xlim((0, len(self.y)))

		# Plot trend
		ax = fig.add_subplot(4, 1, 2)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_ylabel("Trend")
		ax.plot(x, self.trend)
		ax.set_xlim((0, len(self.y)))

		# Plot seasonal
		ax = fig.add_subplot(4, 1, 3)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_ylabel("Seasonal")
		ax.plot(x, self.seasonal)
		ax.set_xlim((0, len(self.y)))

		# Plot seasonal
		ax = fig.add_subplot(4, 1, 4)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_ylabel("Residual")
		ax.scatter(x, self.residual, s = 3)
		ax.plot(x, [0] * len(self.y), c = "r")
		ax.set_xlim((0, len(self.y)))
		plt.show()

def test():
	data = pd.read_csv("data/ndvi.csv")
	ndvi_freq = 24
	St = STL(ndvi_freq, np.array(data["ndvi"])[:510], periodic=True)
	print(St.residual)


if __name__ == '__main__':
	test()