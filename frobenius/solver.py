import numpy as np

from frobenius.apoly import ArrayPoly, det, trim
from frobenius.smith import smith
from math import factorial


def solve(mxA, min_terms=3, lambda_roots=None, atol=1e-12):
    mxL = _calcL(mxA, atol)
    lam = _prepareLambda(mxL[0], lambda_roots=lambda_roots)
    coefsNum = (lam[-1] + min_terms - lam + 0.5).astype(int)
    for n in range(len(mxL), max(coefsNum)):
        mxL.append(ArrayPoly(np.zeros_like(mxA[:1, 0])))
    solutions = []
    for j in range(len(lam)):
        g = _calcCoefs(mxL, lam[j], coefsNum[j], atol)
        solutions.append((lam[j], g))
    return solutions


def _calcL(mxA, atol):
    assert(mxA.ndim == 4)
    mxL = []
    for m in range(mxA.shape[1]):
        mxL.append(trim(ArrayPoly(mxA[:, m]), atol=atol))
    return mxL


def _prepareLambda(mxL0, lambda_roots=None):
    if lambda_roots is None:
        detL0 = det(mxL0)
        lambda_roots = np.roots(detL0.coefs.reshape(-1)[::-1])
    lam = np.sort(np.array(lambda_roots))
    return lam


def _calcCoefs(mxL, lj, coefsNum, atol):
    mxSize = mxL[0].shape[0]
    factor = ArrayPoly([-lj, 1])
    dtype = np.common_type(mxL[0].coefs, factor.coefs)
    mxX, mxY, kappa = \
        _calcSmithN(mxL[0], num=coefsNum, factor=factor, atol=atol)
    alpha, beta = _calcAlphaBeta(kappa)
    mxXt = _calcXt(p=factor, x=mxX, kappa=kappa)
    mxLt = _calcLt(mxL=mxL, p=factor, beta=beta)
    kernelSize = sum(k > 0 for k in kappa[0])
    jordanChainsLen = kappa[0][-kernelSize:]
    del mxX, factor, kappa
    # TODO - move calculation of ct to separate function
    ct = []
    for k in range(kernelSize):
        ctk = np.zeros((coefsNum, alpha + jordanChainsLen[k], mxSize, 1),
            dtype=dtype)
        for n in range(coefsNum):
            nDeriv = beta[n] + jordanChainsLen[k]
            if n == 0:
                b = [_ort(mxSize, -1 - kernelSize + k + 1)]
            else:
                b = _calcB(mxY[n], mxLt[n], lj, ctk[:n, :nDeriv])
            ctk[n, :nDeriv] = _calcCt(mxXt[n], lj, b, nDeriv)
            del nDeriv, b
        ct.append(ctk)
        del ctk
    del mxY, mxXt, mxLt
    # TODO - move calculation of g to separate function
    g = []
    for k in range(kernelSize):
        gk = []
        for q in range(jordanChainsLen[k]):
            gkq = np.zeros((coefsNum, alpha + q + 1, mxSize, 1), dtype=complex)
            for n in range(coefsNum):
                num_ln_terms = beta[n] + q
                pow_deriv_coef = factorial(alpha + q) / factorial(num_ln_terms)
                for m in range(num_ln_terms + 1):
                    gkq[n, m] = pow_deriv_coef * ct[k][n, num_ln_terms - m]
                    pow_deriv_coef *= (num_ln_terms - m) / (m + 1)
                del num_ln_terms, pow_deriv_coef
            while gkq.shape[1] > 1 and np.max(np.abs(gkq[:, -1])) < atol:
                gkq = gkq[:, :-1]
            gk.append(gkq)
            del gkq
        g.append(gk)
        del gk
    return g


def _calcSmithN(a, num, factor, atol):
    mxX = []
    mxY = []
    kappa = []
    for n in range(num):
        lamPlusN = ArrayPoly([n, 1])
        x, y, k = smith(a(lamPlusN), factor, atol=atol)
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
    mxXt = [x[0]]
    for n in range(1, len(x)):
        mxXt.append(ArrayPoly(x[n].coefs.copy()))
        for i in range(len(kappa[n])):
            mxXt[n][:, i] *= p ** (kappa[n][-1] - kappa[n][i])
    return mxXt


def _calcLt(mxL, p, beta):
    mxLt = [[]]
    for n in range(1, len(beta)):
        mxLt.append([])
        for m in range(n):
            factor = p ** (beta[n - 1] - beta[m])
            lamPlusM = ArrayPoly([m, 1])
            mxLt[n].append(trim(factor * mxL[n - m](lamPlusM)))
    return mxLt


def _ort(n, i):
    ort = np.zeros(n)
    ort[i] = 1
    return ort.reshape(-1, 1)


def _calcCt(x, lj, b, nTerms):
    result = []
    for s in range(nTerms):
        result.append(_derivMatMul(x, lj, b, deriv=s))
    return np.array(result)


def _calcB(mxY, mxLt, lj, ct):
    b = []
    inv_left_matrix = np.linalg.inv(mxY(lj))
    for t in range(ct.shape[1]):
        right_part = sum(
            _derivMatMul(mxLt[m], lj, ct[m], deriv=t) \
            for m in range(ct.shape[0]))
        if len(b) > 0:
            # calculate the t-th derivative of mxY @ b
            # except the term with the t-th derivative of b
            right_part += _derivMatMul(mxY, lj, b, deriv=t)
        b.append(-inv_left_matrix @ right_part)
    return b


def _derivMatMul(a, x, b, deriv):
    terms = []
    binom_coef = 1
    # all missing derivatives in b are assumed to be zero
    for t in range(0, min(deriv + 1, len(b))):
        terms.append(binom_coef * a(x, deriv=deriv-t) @ b[t])
        binom_coef *= (deriv - t) / (t + 1)
    return sum(terms)
