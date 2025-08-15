# -*- coding: utf-8 -*-
# @Date:   2024-08-05 22:54:07
# @Last Modified time: 2024-08-05 23:03:44
# @All Rights Reserved!

from functools import partial
import numpy as np
import statsmodels.api as sm

def _no_nans(arr):
	return not np.isnan(arr).any()

def _Xinv0(x, coeffs):
	"""
	Approximate (X'X)^-1 using QR decomposition
	"""
	ncol = np.shape(x)[1]
	r = np.linalg.qr(x)[1]
	qr_rank = np.linalg.matrix_rank(r)

	r = r[:qr_rank, :qr_rank]

	k = coeffs.shape[0]
	rval = np.zeros((k, k))
	rval[:qr_rank, :qr_rank] = np.linalg.inv(r.T @ r)
	return rval

# 计算递归残差
def recresid(x, y, start=None, end=None, tol=None):
	if np.ndim(x) == 1:
		ncol = 1
		nrow = x.shape
	else:
		nrow, ncol = x.shape

	if start is None:
		start = ncol + 1
	if end is None:
		end = nrow
	if tol is None:
		tol = np.sqrt(np.finfo(float).eps / ncol)

	# checks and data dimensions
	assert start > ncol and start <= nrow
	assert end >= start and end <= nrow

	n = end
	q = start - 1
	k = ncol
	rval = np.zeros(n - q)

	# initialize recursion
	y1 = y[:q]

	x_q = x[:q]

	model = sm.OLS(y1, x_q, missing='drop').fit()
	coeffs = model.params

	X1 = _Xinv0(x_q, coeffs)
	betar = np.nan_to_num(coeffs)

	xr = x[q]
	fr = 1 + (xr @ X1 @ xr)
	rval[0] = (y[q] - xr @ betar) / np.sqrt(fr)

	# check recursion against full QR decomposition?
	check = True

	if (q + 1) >= n:
		return rval

	for r in range(q + 1, n):
		# check for NAs in coefficients
		nona = _no_nans(coeffs)

		# recursion formula
		X1 = X1 - (X1 @ np.outer(xr, xr) @ X1)/fr
		betar += X1 @ xr * rval[r-q-1] * np.sqrt(fr)

		# full QR decomposition
		if check:
			y1 = y[:r]
			x_i = x[:r]
			model = sm.OLS(y1, x_i, missing='drop').fit()
			coeffs = model.params
			nona = nona and _no_nans(betar) and _no_nans(coeffs)

			check = not (nona and np.allclose(coeffs, betar, atol=tol))
			X1 = _Xinv0(x_i, coeffs)
			betar = np.nan_to_num(coeffs)

		# residual
		xr = x[r]
		fr = 1 + xr @ X1 @ xr
		val = np.nan_to_num(xr * betar)
		v = (y[r] - np.sum(val)) / np.sqrt(fr)
		rval[r-q] = v

	rval = np.around(rval, 8)
	return rval

# the triangular matrix of Sums of Squared Residuals
def SSRi(i, n, h, X, y, k, intercept_only):
	# 计算第i行的SSR diagonal matrix
	ssr = recresid(X[i:], y[i:])
	rval = np.concatenate((np.repeat(np.nan, k), np.cumsum(ssr**2)))
	return rval

def ssr_triang(n, h, X, y, k, intercept_only):
	my_SSRi = partial(SSRi, n=n, h=h, X=X, y=y, k=k, intercept_only=intercept_only)
	return np.array([my_SSRi(i) for i in range(n-h+1)], dtype=object)

def test():
	x = np.arange(1,21)
	y = x * 2
	print(y)
	y[9:] = y[9:] + 10
	X = sm.add_constant(x)

	rec_res = recresid(X, y)
	print(rec_res)

if __name__ == "__main__":
	test()