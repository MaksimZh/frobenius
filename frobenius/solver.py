import numpy as np

from frobenius.apoly import ArrayPoly, det
from frobenius.smith import smith


def solve(mxA, min_terms=3):
    assert(mxA.ndim == 4)
    mxL = []
    for m in range(mxA.shape[2]):
        mxL.append(ArrayPoly(mxA[:, m]))
    detL0 = det(mxL[0])
    lam = np.sort(np.roots(detL0.coefs.reshape(-1)[::-1]))
    coefsNum = (lam[-1] + min_terms - lam + 0.5).astype(int)
    for j in range(len(lam)):
        _calcCoefs(mxL, lam[j], coefsNum[j])
    return lam


def _calcCoefs(mxL, lj, coefsNum):
    mxSize = mxL[0].shape[0]
    lamMinusLj = ArrayPoly([-lj, 1])
    mxX = []
    mxY = []
    kappa = []
    beta = []
    for n in range(coefsNum):
        lamPlusN = ArrayPoly([n, 1])
        x, y, k = smith(mxL[0](lamPlusN), lamMinusLj)
        mxX.append(x)
        mxY.append(y)
        kappa.append(k)
        beta.append(k[-1])
    alpha = sum(beta[1:])
    kernelSize = sum(k > 0 for k in kappa[0])
    jordanChainsLen = kappa[0][-1 - kernelSize :]
    ct = np.zeros((kernelSize, coefsNum, max(jordanChainsLen)), dtype=complex)
    for k in range(kernelSize):
        for d in range(jordanChainsLen[k]):
            ct[k, 0, d] = _ort(mxSize, -1 - kernelSize + k + 1)


def _ort(n, i):
    ort = np.zeros(n)
    ort[i] = 1
    return ort
