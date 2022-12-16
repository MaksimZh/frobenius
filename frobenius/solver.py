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
        g = _calcCoefs(mxL[:coefsNum[j]], lam[j], atol)
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


def _calcCoefs(mxL, lj, atol):
    ct = _calcAllCt(mxL, lj, atol)
    return [_calcG(ctk, atol) for ctk in ct]


def _calcAllCt(mxL, lj, atol):
    factor = ArrayPoly([-lj, 1])
    ct = _calcInitialCt(mxL[0], lj, atol)
    beta = [0]
    for n in range(1, len(mxL)):
        mxX, mxY, kappa = smith(mxL[0](ArrayPoly([n, 1])), factor, atol=atol)
        mxZ = [trim(factor**(beta[n - 1] - beta[m]) * mxL[n - m](ArrayPoly([m, 1]))) \
            for m in range(n)]
        beta.append(beta[-1] + kappa[-1])
        for k in range(len(ct)):
            nDeriv = len(ct[k][-1]) + kappa[-1]
            b = _calcB(mxY, mxZ, lj, ct[k], nDeriv)
            mxXt = _calcXt(factor, mxX, kappa)
            ct[k].append(_calcCt(mxXt, lj, b, nDeriv))
            del nDeriv, b, mxXt
        del mxX, mxY, kappa
    return ct


def _calcInitialCt(mxL0, lj, atol):
    factor = ArrayPoly([-lj, 1])
    mxX, mxY, kappa = smith(mxL0, factor, atol=atol)
    del factor, mxY
    kernelSize = sum(ka > 0 for ka in kappa)
    mxSize = mxL0.shape[0]
    jordanChainsLen = kappa[-kernelSize:]
    return [[[mxX(lj, deriv=t) @ _ort(mxSize, -1 - kernelSize + k + 1) \
        for t in range(jordanChainsLen[k])]] \
        for k in range(kernelSize)]


def _calcG(ct, atol):
    mxSize = len(ct[0][0])
    coefsNum = len(ct)
    alpha = len(ct[-1]) - len(ct[0])
    g = []
    jordanChainsLen = len(ct[0])
    for q in range(jordanChainsLen):
        gq = np.zeros((coefsNum, alpha + q + 1, mxSize, 1), dtype=complex)
        for n in range(coefsNum):
            beta = len(ct[n]) - jordanChainsLen
            num_ln_terms = beta + q
            pow_deriv_coef = factorial(alpha + q) / factorial(num_ln_terms)
            for m in range(num_ln_terms + 1):
                gq[n, m] = pow_deriv_coef * ct[n][num_ln_terms - m]
                pow_deriv_coef *= (num_ln_terms - m) / (m + 1)
            del beta, num_ln_terms, pow_deriv_coef
        while gq.shape[1] > 1 and np.max(np.abs(gq[:, -1])) < atol:
            gq = gq[:, :-1]
        g.append(gq)
        del gq
    return g


def _calcXt(p, x, kappa):
    mxXt = ArrayPoly(x.coefs.copy())
    for i in range(len(kappa)):
        mxXt[:, i] *= p ** (kappa[-1] - kappa[i])
    return mxXt



def _ort(n, i):
    ort = np.zeros(n)
    ort[i] = 1
    return ort.reshape(-1, 1)


def _calcCt(x, lj, b, nTerms):
    result = []
    for s in range(nTerms):
        result.append(_derivMatMul(x, lj, b, deriv=s))
    return np.array(result)


def _calcB(mxY, mxLt, lj, ct, nDeriv):
    b = []
    inv_left_matrix = np.linalg.inv(mxY(lj))
    for t in range(nDeriv):
        right_part = sum(
            _derivMatMul(mxLt[m], lj, ct[m], deriv=t) \
            for m in range(len(ct)))
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
