# -*- coding: utf-8 -*-

# @Date:   2024-08-05 22:54:07
# @Last Modified time: 2024-08-05 23:03:29
# @All Rights Reserved!

import numpy as np
import pandas as pd
import statsmodels.api as sm
from stl import STL
from efp import EFP
from breakpoints import Breakpoints
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

IS_TEST = False

def partition_matrix(part, mat):
	"""
	Create a partition matrix, given a partition vector and a matrix
	"""
	if part.shape[0] != mat.shape[0]:
		raise ValueError("Partition length must equal Matrix nrows")
	if mat.ndim != 2:
		raise TypeError("mat must be a 2D matrix")

	n_rows, n_cols = mat.shape
	# number of partitions
	n_parts = part[-1] + 1
	ret_val = np.zeros((n_rows, n_parts * n_cols)).astype(float)
	for j in range(n_parts):
		ret_val[(part == j), (j * n_cols):((j + 1) * n_cols)] = mat[part == j, :]
	return ret_val

class BFASTResult():
	def __init__(self, Tt, St, Nt, Vt_bp, Wt_bp, Tt_p_value):
		self.trend = Tt
		self.season = St
		self.remainder = Nt
		self.Tt_p_value = Tt_p_value
		if (Vt_bp == np.array([0])).all():
			self.trend_breakpoints = None
		else:
			self.trend_breakpoints = Vt_bp
		if Wt_bp == np.array([0]).all():
			self.season_breakpoints = None
		else:
			self.season_breakpoints = Wt_bp

	def __str__(self):
		st = "Trend:\n{}\n\n".format(self.trend) +\
			"Season:\n{}\n\n".format(self.season) +\
			"Remainder:\n{}\n\n".format(self.remainder) +\
			"Trend Breakpoints:\n{}\n\n".format(self.trend_breakpoints) +\
			"Season Breakpoints:\n{}\n".format(self.season_breakpoints)
		return st

class Bfast(object):
	"""docstring for Bfast"""
	def __init__(self, data, time, frequency, season = "harmonic", h = 0.15, max_iter = 10, max_breaks = None, level = 0.05):

		# 数据行数
		Yt = data
		nrow = data.shape[0]
		if season == "harmonic":
			f = frequency

			# multiple linear harmonic regression model
			w = 1/f
			tl = np.arange(1, nrow + 1)
			co = np.cos(2 * np.pi * tl * w)
			si = np.sin(2 * np.pi * tl * w)
			co2 = np.cos(2 * np.pi * tl * w * 2)
			si2 = np.sin(2 * np.pi * tl * w * 2)
			co3 = np.cos(2 * np.pi * tl * w * 3)
			si3 = np.sin(2 * np.pi * tl * w * 3)
			smod = np.column_stack((co, si, co2, si2, co3, si3))

			# 季节趋势分解，得到季节项
			St = STL(Yt, f, periodic=True).seasonal
		else:
			St = np.zeros(nrow)
		
		# 去除季节项后数据的break points没用Vt表示
		Vt_bp = np.array([0])
		# 去除趋势项后数据的break points没用Wt表示
		Wt_bp = np.array([0])
		# 上一次迭代的值
		last_Vt_bp = np.array([1])
		last_Wt_bp = np.array([1])
		i = 1
		self.output = None

		while ((Vt_bp != last_Vt_bp).any() or (Wt_bp != last_Wt_bp).any()) and i < max_iter:
			last_Vt_bp = Vt_bp
			last_Wt_bp = Wt_bp

			Vt = Yt - St
			xi = sm.add_constant(time) # 线性模型加上常数项
			# print(xi.shape)
			Tt_stat, Tt_p_value = EFP(xi, Vt, h).sctest()
			# print("Tt_p_value", Tt_p_value)
			Vt_has_bp = False
			# 如果p值小于0.05
			if Tt_p_value < level:
				# 计算break points
				# print(xi, Vt)
				bp_Vt = Breakpoints(xi, Vt, h = h, max_breaks = max_breaks)
				if bp_Vt.breakpoints is not None:
					Vt_has_bp = True
				else:
					Vt_has_bp = False


			if not Vt_has_bp:
				# 没有break point时
				fm0 = sm.OLS(Vt, time, missing='drop').fit()
				Tt_p_value = fm0.pvalues[0]
				if IS_TEST:
					print("------Tt_p_value", fm0.pvalues)
				Vt_bp = np.array([0])
				Tt = fm0.predict(exog=time)
			else:
				part = bp_Vt.breakfactor()
				X1 = partition_matrix(part, xi)
				fm1 = sm.OLS(Vt, X1, missing='drop').fit()
				Vt_bp = bp_Vt.breakpoints
				Tt = fm1.predict()

			# print("Tt", Tt)


			if season == "none":
				Wt = np.zeros(nrow).astype(float)
				St = np.zeros(nrow).astype(float)
			else:
				# 季节项
				Wt = Yt - Tt
				Wt_stat, Wt_p_value = EFP(smod, Wt, h).sctest()
				Wt_has_bp = False
				if Wt_p_value < level:
					bp_Wt = Breakpoints(smod, Wt, h=h, max_breaks=max_breaks)
					if bp_Wt.breakpoints is not None:
						Wt_has_bp = True
					else:
						Wt_has_bp = False

				if not Wt_has_bp:
					sm0 = sm.OLS(Wt, smod, missing='drop').fit()
					St = sm0.predict()
					Wt_bp = np.array([0])
				else:
					part = bp_Wt.breakfactor()
					X_sm1 = partition_matrix(part, smod1)
					sm1 = sm.OLS(Wt1, X_sm1, missing='drop').fit()
					Wt_bp = bp_Wt.breakpoints
					St = sm1.predict()

			Nt = Yt - Tt - St
			i = i + 1
			# print(Vt_bp)
			self.output = BFASTResult(Tt, St, Nt, Vt_bp, Wt_bp, Tt_p_value)

def read_data():
	path = "data/harvest.csv"
	data = pd.read_csv(path)
	# print(data["harvest"])
	return data

# 根据分割点进行画图
def segmented_plot(ax, x, arr, bp):
	prev = 0
	vals = np.concatenate((bp, [arr.shape[0] - 1]))
	for i, s in enumerate(vals + 1):
		ind = max(0, prev-1)
		end_ind = s - 1
		if i == len(vals) - 1:
			end_ind = s
		ax.plot(x[ind:end_ind], arr[ind:end_ind], label="seg {}".format(i+1))
		ax.legend()
		prev = s

def get_trend_type(vo):
	Tt  = vo.trend
	Tt_bp = vo.trend_breakpoints
	Tt_p_value = vo.Tt_p_value
	break_year = 0

	if Tt_bp is None:
		if Tt_p_value > 0.05:
			return 8, break_year
		elif Tt[0] > Tt[-1]:
			# 单调型减
			return 2, break_year
		else:
			# 单调型加
			return 0, break_year

	arr, bp = Tt, Tt_bp

	prev = 0
	vals = np.concatenate((bp, [arr.shape[0] - 1]))
	datas = []
	for i, s in enumerate(vals + 1):
		ind = max(0, prev-1)
		end_ind = s - 1
		if i == len(vals) - 1:
			end_ind = s
		if i == 1:
			break_year = ind
		data = arr[ind:end_ind]
		datas.append(data)
		prev = s
	if len(datas) == 1:
		data = datas[0]
		if Tt_p_value > 0.05:
			return 8, break_year
		elif data[0] > data[-1]:
			# 单调型减
			return 2, break_year
		else:
			# 单调型加
			return 0, break_year
	elif len(datas) == 2:
		data0 = datas[0]
		data1 = datas[1]
		if data0[0] > data0[-1] and data1[0] > data1[-1] and data0[-1] > data1[0]:
			# 单调型减
			return 3, break_year
		elif data0[0] < data0[-1] and data1[0] < data1[-1] and data0[-1] < data1[0]:
			# 单调型加
			return 1, break_year
		elif data0[0] < data0[-1] and data1[0] < data1[-1] and data0[-1] > data1[0]:
			# 中断型加
			return 4, break_year
		elif data0[0] > data0[-1] and data1[0] > data1[-1] and data0[-1] < data1[0]:
			# 中断型减
			return 5, break_year
		elif data0[0] < data0[-1] and data1[0] > data1[-1]:
			# 由增到减
			return 6, break_year
		elif data0[0] > data0[-1] and data1[0] < data1[-1]:
			# 由减到增加
			return 7, break_year
		else:
			return 9, break_year
	else:
		# 变动激烈
		return 10, break_year

def plot(x, y, vo):
	figsz = (16, 10)

	fig = plt.figure(figsize=figsz)
	ax = fig.add_subplot(4, 1, 1)

	Tt  = vo.trend
	St = vo.season
	Rt = vo.remainder
	Tt_bp = vo.trend_breakpoints
	St_bp = vo.season_breakpoints
	# plot Y
	ax.set_title("observations")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	if Tt_bp is not None:
		segmented_plot(ax, x, Tt, Tt_bp)
	else:
		ax.plot(x, Tt)
	ax.plot(x, y)

	# plot trend
	ax = fig.add_subplot(4, 1, 2)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_title("trend")
	if Tt_bp is not None:
		segmented_plot(ax, x, Tt, Tt_bp)
	else:
		ax.plot(x, Tt)

	# plot season
	ax = fig.add_subplot(4, 1, 3)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_title("season")
	if St_bp is not None:
		segmented_plot(ax, x, St, St_bp)
	else:
		ax.plot(x, St)

	# plot remainder
	ax = fig.add_subplot(4, 1, 4)
	ax.set_title("remainder")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.plot(x, Rt)
	plt.show()

def main():
	data = read_data()
	# print(data)
	frequency = 23
	level = 0.05
	h = 0.15
	max_iter = 10
	vo = Bfast(np.array(data["harvest"]), np.array(data["dates"]), frequency, level=level, h=h, max_iter=max_iter, max_breaks = None).output
	# print("vo", vo, len(vo.trend_breakpoints))
	print(len(vo.trend_breakpoints))
	Tt  = vo.trend
	Tt_bp = vo.trend_breakpoints
	print("type:", get_trend_type(vo))
	plot(np.array(data["dates"]), np.array(data["harvest"]), vo)

def bfast_main(x, y):
	frequency = 1
	level = 0.05
	h = 0.20
	max_iter = 10
	data = np.array(y)
	time = np.array(x)
	best_h = h
	min_p_value = 1.0
	for item_h in [0.19, 0.20, 0.25, 0.3, 0.35, 0.4]:
			xi = sm.add_constant(time) # 线性模型加上常数项
			# print(xi.shape)
			Tt_stat, Tt_p_value = EFP(xi, data, item_h).sctest()
			if Tt_p_value < level:
				best_h = item_h
				if IS_TEST:
					print("found best h:", best_h)
				break

			if min_p_value > Tt_p_value:
				min_p_value = Tt_p_value
				best_h = item_h

	if IS_TEST:
		print("best h:", best_h)
	vo = Bfast(np.array(y), np.array(x), frequency, season = "none", level=level, h = best_h, max_iter=max_iter, max_breaks = 1).output
	# print("vo", vo)
	if IS_TEST:
		print("vo", vo.Tt_p_value)
		plot(np.array(x), np.array(y), vo)

	if vo.trend_breakpoints is None:
		bp_cnt = 0
	else:
		bp_cnt = len(vo.trend_breakpoints)
		# plot(np.array(x), np.array(y), vo)
	bp_type, break_year = get_trend_type(vo)
	if bp_cnt == 0 and bp_type == 2 and IS_TEST:
		plot(np.array(x), np.array(y), vo)
	return bp_cnt, bp_type, break_year


if __name__ == '__main__':
	main()

				

