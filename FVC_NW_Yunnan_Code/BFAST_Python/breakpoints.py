# -*- coding: utf-8 -*-
# @Date:   2024-08-05 22:54:07
# @Last Modified time: 2024-08-05 23:03:34
# @All Rights Reserved!

import numpy as np
np.set_printoptions(precision = 2, linewidth = 320)

import matplotlib.pyplot as plt
import pandas as pd
import ssr

class Breakpoints(object):
    """docstring for Breakpoints"""
    def __init__(self, X, y, h = 0.15, max_breaks = None):
        n, k = X.shape
        self.nobs = n

        self.h = int(np.floor(n * h))
        intercept_only = np.allclose(X, 1)

        # 允许最多的break points个数
        max_allowed_breaks = int(np.ceil(n / self.h) - 2)
        if max_breaks is None:
            max_breaks = max_allowed_breaks
        elif max_breaks > max_allowed_breaks:
            max_breaks = max_allowed_breaks
        self.max_breaks = max_breaks

        # 计算triangular matrix的Sums of Squared Residuals
        self.SSR_triang = ssr.ssr_triang(n, self.h, X, y, k, intercept_only)
        index = np.arange((self.h - 1), (n - self.h)).astype(int)

        ## 1 break
        break_SSR = np.array([self.SSR(0, i) for i in index])
        SSR_table = np.column_stack((index, break_SSR))

        ## breaks >= 2
        SSR_table = self.extend_SSR_table(SSR_table, self.max_breaks)

        opt = self.extract_breaks(SSR_table, self.max_breaks).astype(int)

        # print("SSR_table", SSR_table)
        self.SSR_table = SSR_table
        self.nreg = k
        self.y = y
        self.X = X

        # 提取break points
        _, BIC_table = self.summary()
        breaks = np.argmin(BIC_table)
        _, bp = self.breakpoints_for_m(breaks)
        if bp is None:
            self.breakpoints_no_nans = bp
            self.breakpoints = bp
        else:
            self.breakpoints_no_nans = bp.astype(int)
            self.breakpoints = bp.astype(int)

    def SSR(self, i, j):
        # table lookup
        return self.SSR_triang[int(i)][int(j - i)]

    def extend_SSR_table(self, SSR_table, breaks):
        _, ncol = SSR_table.shape
        h = self.h
        n = self.nobs

        if (breaks * 2) > ncol:
            v1 = int(ncol/2) + 1
            v2 = breaks

            if v1 < v2:
                loop_range = np.arange(v1, v2 + 1)
            else:
                loop_range = np.arange(v1, v2 - 1, -1)

            for m in loop_range:
                my_index = np.arange((m * h) - 1, (n - h))
                index_arr = np.arange((m - 1) * 2 - 2, (m - 1) * 2)
                my_SSR_table = SSR_table[:, index_arr]
                nans = np.repeat(np.nan, my_SSR_table.shape[0])
                my_SSR_table = np.column_stack((my_SSR_table, nans, nans))
                for i in my_index:
                    pot_index = np.arange((m - 1) * h - 1, (i - h + 1)).astype(int)
                    fun = lambda j: my_SSR_table[j - h + 1, 1] + self.SSR(j + 1, i)
                    # map
                    break_SSR = np.vectorize(fun)(pot_index)
                    opt = np.nanargmin(break_SSR)
                    my_SSR_table[i - h + 1, np.array((2, 3))] = \
                        np.array((pot_index[opt], break_SSR[opt]))

                SSR_table = np.column_stack((SSR_table, my_SSR_table[:, np.array((2,3))]))
        return SSR_table

    def extract_breaks(self, SSR_table, breaks):
        """
        extract optimal breaks
        """
        _, ncol = SSR_table.shape
        n = self.nobs
        h = self.h

        if breaks * 2 > ncol:
            raise ValueError("compute SSR_table with enough breaks before")

        index = SSR_table[:, 0].astype(int)
        fun = lambda i: SSR_table[int(i - self.h + 1), int(breaks * 2 - 1)] \
            + self.SSR(i + 1, n - 1)
        # parallel map
        break_SSR = np.vectorize(fun)(index)
        opt = np.zeros(breaks, dtype=int)
        opt[-1] = index[np.nanargmin(break_SSR)]
        if breaks > 1:
            # sequential(!) fold
            for j in np.arange(breaks - 2, -1, -1).astype(int):
                i = 2 * (j + 1)
                opt[j] = SSR_table[int(opt[j + 1] - h + 1), i]
        return np.array(opt)

    def summary(self):
        """
        Calculates Sums of Squared Residuals and BIC for m in 0..max_breaks
        """
        n = self.nobs
        SSR = np.concatenate(([self.SSR(0, n - 1)], np.repeat(np.nan, self.max_breaks)))
        if np.isclose(SSR[0], 0.0):
            BIC_val = -np.inf
        else:
            BIC_val = n * (np.log(SSR[0]) + 1 - np.log(n) + np.log(2 * np.pi)) \
                + np.log(n) * (self.nreg + 1)
        BIC = np.concatenate(([BIC_val], np.repeat(np.nan, self.max_breaks)))
        SSR1, breakpoints = self.breakpoints_for_m(self.max_breaks)
        SSR[self.max_breaks] = SSR1
        BIC[self.max_breaks] = self.BIC(SSR1, breakpoints)

        if self.max_breaks > 1:
            # parallel map
            for m in range(1, self.max_breaks):
                SSR_m, breakpoints_m = self.breakpoints_for_m(m)
                SSR[m] = SSR_m
                BIC[m] = self.BIC(SSR_m, breakpoints_m)
        retval = np.vstack((SSR, BIC))
        return retval

    def breakpoints_for_m(self, m):
        if m < 1:
            SSR = self.SSR(0, self.nobs - 1)
            return SSR, None
        else:
            breakpoints = self.extract_breaks(self.SSR_table, m)
            # map reduce
            bp = np.concatenate(([0], breakpoints, [self.nobs-1]))
            cb = np.column_stack((bp[:-1] + 1, bp[1:]))
            fun = lambda x: self.SSR(x[0], x[1])
            SSR = np.sum([fun(i) for i in cb])
            return SSR, breakpoints

    def BIC(self, SSR, breakpoints):
        """
        Bayesian Information Criterion
        """
        # scalar
        if np.isclose(SSR, 0.0):
            return -np.inf
        n = self.nobs
        df = (self.nreg + 1) * (len(breakpoints[~np.isnan(breakpoints)]) + 1)
        # log-likelihood
        logL = n * (np.log(SSR) + 1 - np.log(n) + np.log(2 * np.pi))
        bic = df * np.log(n) + logL
        return bic

    def breakfactor(self):
        breaks = self.breakpoints
        nobs = self.nobs
        if np.isnan(breaks).all():
            return np.repeat(1, nobs)

        nbreaks = breaks.shape[0]
        # scan
        v = np.insert(np.diff(np.append(breaks, nobs)), 0, breaks[0]).astype(int)
        fac = np.repeat(np.arange(1, nbreaks + 2), v)
        return fac - 1

if __name__ == '__main__':
    n = 50
    ones = np.ones(n).reshape((n, 1)).astype("float64")
    y = np.arange(1, n+1).astype("float64")
    X = np.copy(y).reshape((n, 1))
    y[14:] = y[14:] * 0.03
    y[34:] = y[34:] + 10

    plt.plot(X, y)
    plt.show()

    bp = Breakpoints(X, y).breakpoints
    print("Breakpoints:", bp)