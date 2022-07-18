import numpy as np

from frobenius.apoly import ArrayPoly, det, trim
from frobenius.smith import smith
from math import factorial


def solve(mxA, min_terms=3, lambda_roots=None, atol=1e-12):
    mxL = _calcL(mxA)
    lam = _prepareLambda(mxL[0], lambda_roots=lambda_roots)
    coefsNum = (lam[-1] + min_terms - lam + 0.5).astype(int)
    for n in range(len(mxL), max(coefsNum)):
        mxL.append(ArrayPoly(np.zeros_like(mxA[:1, 0])))
    result = []
    for j in range(len(lam)):
        g = _calcCoefs(mxL, lam[j], coefsNum[j], atol)
        result.append((lam[j], g))
    return result


def _calcL(mxA):
    assert(mxA.ndim == 4)
    mxL = []
    for m in range(mxA.shape[1]):
        mxL.append(ArrayPoly(mxA[:, m]))
    return mxL


def _prepareLambda(mxL0, lambda_roots=None):
    if lambda_roots is None:
        detL0 = det(mxL0)
        lambda_roots = np.roots(detL0.coefs.reshape(-1)[::-1])
    lam = np.sort(np.array(lambda_roots))
    return lam


def _calcCoefs(mxL, lj, coefsNum, atol):
    mxSize = mxL[0].shape[0]
    lamMinusLj = ArrayPoly([-lj, 1])
    dtype = np.common_type(mxL[0].coefs, lamMinusLj.coefs)
    mxX, mxY, kappa = \
        _calcSmithN(mxL[0], num=coefsNum, p=lamMinusLj, atol=atol)
    alpha, beta = _calcAlphaBeta(kappa)
    mxXt = _calcXt(p=lamMinusLj, x=mxX, kappa=kappa)
    mxLt = _calcLt(mxL=mxL, p=lamMinusLj, beta=beta)
    kernelSize = sum(k > 0 for k in kappa[0])
    jordanChainsLen = kappa[0][-1 - kernelSize :]
    ct = []
    for k in range(kernelSize):
        ctk = np.zeros((coefsNum, alpha + jordanChainsLen[k], mxSize, 1),
            dtype=dtype)
        for d in range(jordanChainsLen[k]):
            ctk[0, d] = \
                mxX[0](lj, deriv=d) @ _ort(mxSize, -1 - kernelSize + k + 1)
        for n in range(1, coefsNum):
            nDeriv = beta[n] + jordanChainsLen[k]
            b = np.zeros((nDeriv, mxSize, 1), dtype=dtype)
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
                            factor * mxLt[n][m](lj, deriv=d-s) @ ctk[m, s]
                        factor *= (d - s) / (s + 1)
                b[d] = -invY @ (sumDeriv + sumPrev)
                factor = 1
                for s in range(d + 1):
                    ctk[n, d] = mxXt[n](lj, deriv=d-s) @ b[s]
                    factor *= (d - s) / (s + 1)
        ct.append(ctk)
    g = []
    for k in range(kernelSize):
        gk = []
        for q in range(jordanChainsLen[k]):
            gkq = np.zeros((coefsNum, alpha + q + 1), dtype=complex)
            for n in range(coefsNum):
                mMax = beta[n] + q
                factor = factorial(alpha + q) / factorial(mMax)
                for m in range(mMax + 1):
                    gkq[n, m] = factor * ct[k][n, mMax - m]
                    factor *= (mMax - m) / (m + 1)
            while gkq.shape[1] > 1 and max(np.abs(gkq[:, -1])) < atol:
                gkq = gkq[:, :-1]
            gk.append(gkq)
        g.append(gk)
    return g


def _calcSmithN(a, num, p, atol):
    mxX = []
    mxY = []
    kappa = []
    for n in range(num):
        lamPlusN = ArrayPoly([n, 1])
        x, y, k = smith(a(lamPlusN), p, atol=atol)
        mxX.append(x)
        mxY.append(y)
        kappa.append(k)
    return mxX, mxY, kappa


def _calcAlphaBeta(kappa):
    beta = [0]
    for ka in kappa[1:]:
        beta.append(beta[-1] + ka[-1])
    alpha = beta[-1]
    return alpha, beta


def _calcXt(p, x, kappa):
    mxXt = [None]
    for n in range(1, len(x)):
        mxXt.append(ArrayPoly(x[n].coefs.copy()))
        for i in range(len(kappa[n])):
            mxXt[n][:, i] *= p ** (kappa[n][-1] - kappa[n][i])
    return mxXt


def _calcLt(mxL, p, beta):
    mxLt = [[]]
    for n in range(1, len(mxL)):
        mxLt.append([])
        for m in range(n):
            factor = p ** (beta[n - 1] - beta[m])
            lamPlusM = ArrayPoly([m, 1])
            mxLt[n].append(trim(factor * mxL[n - m](lamPlusM)))
    return mxLt


def _ort(n, i):
    ort = np.zeros(n)
    ort[i] = 1
    return ort
