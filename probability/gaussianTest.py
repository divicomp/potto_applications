from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
import unittest
from unittest import TestCase
from numpy.random import uniform
import numpy as np
import random
from tqdm import trange
import math

from scipy.special import erf

from potto.diffeos import Shift, FlipShift, SquareSlice, QuarterCircle, Simplex3D
from potto.grammar import (
    CONTEXT,
    Delta,
    Diffeomorphism,
    Measure,
    Var,
    TegVar,
    Heaviside,
    GExpr,
    Const,
    Int,
    Hole,
    Sym,
    Uid,
)
from potto.pmath import Abs, Sqrt, Exp
from potto.derivative import deriv

from potto.evaluate import evaluate_all, evaluate
from potto.samples import Sample, Samples, Support, VarVal
from potto.simplify import simplify

from potto.measure import BoundedLebesgue

sigma_h = 2.0
mu_h = 1.0
denom = math.sqrt(2 * math.pi)


def errorFunction(giv, mean, sigma):
    upper = (giv - mean) / (sigma)
    upper = upper * (0.7071067811865475)  # Magic number is 1/sqrt(2)
    t = TegVar("t")
    integrand = Const(1.1283791670955126) * Exp(-(t**2))  # Magic number is \frac{2}{\sqrt{\pi}}
    r = BoundedLebesgue(Const(0), upper, t)
    i = Int(integrand, r)
    # return integrate(i, num_samples=1000)
    return i


def getLoss(given):
    # Setup of variables

    # Hidden ground truth
    sigma_h = 2.0
    mu_h = 1.0

    # precalc of common used denom
    denom = math.sqrt(2 * math.pi)

    a = Var("a")  # lower bound of trunc gauss
    b = Var("b")  # upper bound of trunc gauss
    sigma = Var("s")  # trunc gauss SD
    mu = Var("m")  # trunc gauss expectation.
    x = TegVar("x")  # Variable of gaussian
    r = BoundedLebesgue(-100, 100, x)  # Sampling distribution for integration.

    erf_a = (1 / 2) * (1 + errorFunction(given[0], given[2], given[3]))
    erf_b = (1 / 2) * (1 + errorFunction(given[1], given[2], given[3]))

    scaling_factor = (1 / sigma) * (1 / (erf_b - erf_a))  # Scalling measure of trunc gaussian to 1
    ground_truth = ((1.0) / (sigma_h * denom)) * Exp(-(1.0 / 2.0) * (((x - mu_h) / sigma_h) ** 2))  # Gaussian

    # scalled gaussian of given guess
    trunc_gauss = scaling_factor * (1) / (denom) * Exp((-1) / (2) * ((x - mu) / (sigma)) ** 2)

    # Arithmetic AND with two heaviside which gives 1 if between a and b and 0 otherwise.
    heaviside_bound = Heaviside(FlipShift((b,), (x,), Const(-1))) * Heaviside(Shift((a,), (x,), Const(1)))

    integrand = trunc_gauss * heaviside_bound

    f_x = Int(integrand, r)
    var_val = VarVal({a.name: given[0], b.name: given[1], mu.name: given[2], sigma.name: given[3]})
    # We sample the outer integral 1000 times
    # each of the inner integrals are sampled 1000 times.
    # In total, we take 2 million samples.
    return evaluate_all(f_x, var_val, num_samples=100)


def variational_inference(base):
    pass


base = np.array([1, 4, 1, 1])

if __name__ == "__main__":
    print(getLoss(base))
