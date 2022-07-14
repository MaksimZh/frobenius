import numpy as np

from frobenius.apoly import ArrayPoly, det
from frobenius.smith import smith


def solve(mxA, min_terms=3):
    assert(mxA.ndim == 4)
    mxL = []
    for m in range(mxA.shape[1]):
        mxL.append(ArrayPoly(mxA[:, m]))
    detL0 = det(mxL[0])
    lam = np.sort(np.roots(detL0.coefs.reshape(-1)[::-1]))
    coefsNum = (lam[-1] + min_terms - lam + 0.5).astype(int)
    for n in range(len(mxL), max(coefsNum)):
        mxL.append(ArrayPoly(np.zeros_like(mxA[:1, 0])))
    for j in range(len(lam)):
        _calcCoefs(mxL, lam[j], coefsNum[j])
    return lam


def _calcCoefs(mxL, lj, coefsNum):
    mxSize = mxL[0].shape[0]
    lamMinusLj = ArrayPoly([-lj, 1])
    mxX = []
    mxY = []
    kappa = []
    for n in range(coefsNum):
        lamPlusN = ArrayPoly([n, 1])
        x, y, k = smith(mxL[0](lamPlusN), lamMinusLj)
        mxX.append(x)
        mxY.append(y)
        kappa.append(k)
    beta = [k[-1] for k in kappa]
    mxXt = [None]
    for n in range(1, coefsNum):
        mxXt.append(ArrayPoly(mxX[n].coefs.copy()))
        for i in range(mxSize):
            mxXt[n][:, i] *= lamMinusLj ** (kappa[n][-1] - kappa[n][i])
    mxLt = [[]]
    for n in range(1, coefsNum):
        mxLt.append([])
        for m in range(n + 1
        ):
            factor = lamMinusLj ** (beta[n - 1] - beta[m])
            lamPlusM = ArrayPoly([m, 1])
            mxLt[n].append(factor * mxL[n - m](lamPlusM))
    alpha = sum(beta[1:])
    kernelSize = sum(k > 0 for k in kappa[0])
    jordanChainsLen = kappa[0][-1 - kernelSize :]
    ct = np.zeros(
        (kernelSize, coefsNum, alpha + max(jordanChainsLen),
            mxSize, 1), 
        dtype=complex)
    for k in range(kernelSize):
        for d in range(jordanChainsLen[k]):
            ct[k, 0, d] = \
                mxX[0](lj, deriv=d) @ _ort(mxSize, -1 - kernelSize + k + 1)
        for n in range(1, coefsNum):
            nDeriv = beta[n] + jordanChainsLen[k]
            b = np.zeros((nDeriv, mxSize, 1), dtype=complex)
            invY = np.linalg.inv(mxY[n](lj))
            for d in range(nDeriv):
                sumDeriv = np.zeros_like(b[0])
                factor = 1
                for s in range(d):
                    sumDeriv += factor * mxY[n](lj, deriv=d-s) @ b[s]
                    factor *= (d - s) / (s + 1)
                sumPrev = np.zeros_like(b[0])
                for m in range(n):
                    factor = 1
                    for s in range(d + 1):
                        sumPrev += \
                            factor * mxLt[n][m](lj, deriv=d-s) @ ct[k, m, s]
                        factor *= (d - s) / (s + 1)
                b[d] = -invY @ (sumDeriv + sumPrev)
                factor = 1
                for s in range(d + 1):
                    ct[k, n, d] = mxXt[n](lj, deriv=d-s) @ b[s]
                    factor *= (d - s) / (s + 1)
    print(ct)


def _ort(n, i):
    ort = np.zeros(n)
    ort[i] = 1
    return ort
