# -*- coding: utf-8 -*-
# @Date:   2024-08-05 22:54:07
# @Last Modified time: 2024-08-05 23:03:39
# @All Rights Reserved!


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Empirical fluctuation process
class EFP(object):
	"""docstring for EFP"""
	def __init__(self, X, y, h):
		'''
		param h: bandwidth parameter for the MOSUM process
		'''
		y = np.array(y)
		n, k = X.shape

		# 最小二乘拟合线性模型
		fm = sm.OLS(y, X, missing='drop').fit()

		# 计算残差
		e = y - fm.predict(exog = X)

		# 计算标准差
		sigma = np.sqrt(np.sum(e**2) / (n - k))

		# 计算一个bandwidthd的个数
		nh = int(np.floor(n * h))
		# print(nh)

		e_zero = np.insert(e, 0, 0)
		# 求累加
		process = np.cumsum(e_zero)
		# 计算每个带宽内的残差和，计算过程可参考论文Strucchange-An R package for testing for structural change in linear regression models.pdf
		process = process[nh:] - process[:n - nh + 1]
		process = process / (sigma * np.sqrt(n))

		self.coefficients = fm
		self.sigma = sigma
		self.e = e
		self.process = process
		self.h = h

	def p_value(self, x, h, n, max_n = 6):
		# 通过查表计算p值
		# 可参考论文The Moving-Estimates Test for Parameter Stability.pdf Table 2.
		sc_me = np.array([0.7552, 0.9809, 1.1211, 1.217, 1.2811, 1.3258, 1.3514, 1.3628,
		 1.361, 1.3751, 0.7997, 1.0448, 1.203, 1.3112, 1.387, 1.4422, 1.4707, 1.4892,
		 1.4902, 1.5067, 0.825, 1.0802, 1.2491, 1.3647, 1.4449, 1.5045, 1.5353, 1.5588,
		 1.563, 1.5785, 0.8414, 1.1066, 1.2792, 1.3973, 1.4852, 1.5429, 1.5852, 1.6057,
		 1.6089, 1.6275, 0.8541, 1.1247, 1.304, 1.425, 1.5154, 1.5738, 1.6182, 1.646,
		 1.6462, 1.6644, 0.8653, 1.1415, 1.3223, 1.4483, 1.5392, 1.6025, 1.6462, 1.6697,
		 1.6802, 1.6939, 0.8017, 1.0483, 1.2059, 1.3158, 1.392, 1.4448, 1.4789, 1.4956,
		 1.4976, 1.5115, 0.8431, 1.1067, 1.2805, 1.4042, 1.4865, 1.5538, 1.59, 1.6105,
		 1.6156, 1.6319, 0.8668, 1.1419, 1.3259, 1.4516, 1.5421, 1.6089, 1.656, 1.6751,
		 1.6828, 1.6981, 0.8828, 1.1663, 1.3533, 1.4506, 1.5791, 1.6465, 1.6927, 1.7195,
		 1.7245, 1.7435, 0.8948, 1.1846, 1.3765, 1.5069, 1.6077, 1.677, 1.7217, 1.754,
		 1.7574, 1.7777, 0.9048, 1.1997, 1.3938, 1.5305, 1.6317, 1.7018, 1.7499, 1.7769,
		 1.7889, 1.8052, 0.8444, 1.1119, 1.2845, 1.4053, 1.4917, 1.5548, 1.5946, 1.6152,
		 1.621, 1.6341, 0.8838, 1.1654, 1.3509, 1.4881, 1.5779, 1.653, 1.6953, 1.7206,
		 1.7297, 1.7455, 0.904, 1.1986, 1.3951, 1.5326, 1.6322, 1.7008, 1.751, 1.7809,
		 1.7901, 1.8071, 0.9205, 1.2217, 1.4212, 1.5593, 1.669, 1.742, 1.7941, 1.8212,
		 1.8269, 1.8495, 0.9321, 1.2395, 1.444, 1.5855, 1.6921, 1.7687, 1.8176, 1.8553,
		 1.8615, 1.8816, 0.9414, 1.253, 1.4596, 1.61, 1.7139, 1.793, 1.8439, 1.8763,
		 1.8932, 1.9074, 0.8977, 1.1888, 1.3767, 1.5131, 1.6118, 1.6863, 1.7339, 1.7572,
		 1.7676, 1.7808, 0.9351, 1.2388, 1.4362, 1.5876, 1.693, 1.7724, 1.8223, 1.8559,
		 1.8668, 1.8827, 0.9519, 1.27, 1.482, 1.6302, 1.747, 1.8143, 1.8756, 1.9105,
		 1.919, 1.9395, 0.9681, 1.2918, 1.5013, 1.6536, 1.7741, 1.8573, 1.914, 1.945,
		 1.9592, 1.9787, 0.9799, 1.3088, 1.5252, 1.6791, 1.7967, 1.8837, 1.9377, 1.9788,
		 1.9897, 2.0085, 0.988, 1.622, 1.5392, 1.7014, 1.8154, 1.9061, 1.9605, 1.9986,
		 2.0163, 2.0326]).reshape((60, 4), order='F')
		n = min(n, max_n)
		table_dim = 10
		# tableh和crit_table是对应的
		crit_table = sc_me[((n - 1) * table_dim):(n * table_dim),:]
		tableh = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
		tablep = np.array((0.1, 0.05, 0.025, 0.01))

		tablen = 4
		tableipl = np.zeros(tablen)
		# 对h进行插值，计算每个p值对应的critical value
		for i in range(tablen):
			tableipl[i] = np.interp(h, tableh, crit_table[:, i])

		tableipl = np.insert(tableipl, 0, 0)
		tablep = np.insert(tablep, 0, 1)

		# x插值得到p值
		p = np.interp(x, tableipl, tablep)
		return p

	def sctest(self, functional="max"):
		h = self.h
		x = self.process
		nd = np.ndim(x)

		# 计算残差绝对值最大值
		stat = np.max(np.abs(x))
		p_value = self.p_value(stat, h, nd)

		return stat, p_value

def test():
	h = 0.15
	data = pd.read_csv("data/nile.csv")
	data_size = len(data)
	y = np.array(data["nile"])
	x = np.ones(data_size).reshape(data_size, 1)
	efp = EFP(x, y, h)
	stat, p_value = efp.sctest()
	print(stat, p_value)

if __name__ == '__main__':
	test()