# -*- coding: utf-8 -*-
"""
Created on Sun May  5 09:52:45 2019

@author: anurag

Implementation of benchmark functions as made in the nevergrad library
https://github.com/facebookresearch/nevergrad

"""

import numpy as np

REGISTRY_OF_FUNCS = dict()
def func_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    REGISTRY_OF_FUNCS[str(func.__name__)] = func
    return func


@func_reg_deco
def sphere(x: np.ndarray) -> float:
    """The most classical continuous optimization testbed.
    
    If you do not solve that one then you have a bug."""
    return float(np.sum(x**2))

@func_reg_deco
def sphere4(x: np.ndarray) -> float:
    """
    Translated sphere function.
    The most classical continuous optimization testbed.
    If you do not solve that one then you have a bug.
    """
    return float(np.sum((x - 4.)**2))

@func_reg_deco
def sumdeceptive(x: np.ndarray) -> float:
    dec = 3 * x**2 - (2 / (3**(x - 2)**2 + .1))
    return float(np.sum(dec))

@func_reg_deco
def cigar(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.
    The other classical example is ellipsoid.
    """
    return float(x[0]**2 + 1000000. * np.sum(x[1:]**2))

@func_reg_deco
def altcigar(x: np.ndarray) -> float:
    """Similar to cigar, but variables in inverse order.
    
    E.g. for pointing out algorithms not invariant to the order of variables."""
    return float(x[-1]**2 + 1000000. * np.sum(x[:-1]**2))

@func_reg_deco
def rastrigin(x: np.ndarray) -> float:
    """Classical multimodal function."""
    cosi = float(np.sum(np.cos(2 * np.pi * x)))
    return float(10 * (len(x) - cosi) + sphere(x))

@func_reg_deco
def hm(x: np.ndarray) -> float:
    """New multimodal function (proposed for Nevergrad)."""
    return float(np.sum((x**2) * (1.1 + np.cos(1. / x))))


@func_reg_deco
def ellipsoid(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.
    The other classical example is cigar.
    """
    return sum((10**(6 * (i - 1) / float(len(x) - 1))) * (x[i]**2) for i in range(len(x)))

@func_reg_deco
def altellipsoid(y: np.ndarray) -> float:
    """Similar to Ellipsoid, but variables in inverse order.
    
    E.g. for pointing out algorithms not invariant to the order of variables."""
    x = y[::-1]
    return sum((10**(6 * (i - 1) / float(len(x) - 1))) * (x[i]**2) for i in range(len(x)))


@func_reg_deco
def rosenbrock(x: np.ndarray) -> float:
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


@func_reg_deco
def griewank(x: np.ndarray) -> float:
    """Multimodal function, often used in Bayesian optimization."""
    part1 = np.sum(x**2)
    part2 = np.prod(np.cos(x / np.sqrt(1 + np.arange(len(x)))))
    return 1 + (float(part1)/4000.0) - float(part2)


@func_reg_deco
def deceptiveillcond(x: np.ndarray) -> float:
    """An extreme ill conditioned functions. Most algorithms fail on this.
    
    The condition number increases to infinity as we get closer to the optimum."""
    assert len(x) >= 2
    return float(max(np.abs(np.arctan(x[1]/x[0])),
                     np.sqrt(x[0]**2. + x[1]**2.),
                     1. if x[0] > 0 else 0.) if x[0] != 0. else float("inf"))


@func_reg_deco
def deceptivepath(x: np.ndarray) -> float:
    """A function which needs following a long path. Most algorithms fail on this.
    
    The path becomes thiner as we get closer to the optimum."""
    assert len(x) >= 2
    distance = np.sqrt(x[0]**2 + x[1]**2)
    if distance == 0.:
        return 0.
    angle = np.arctan(x[0] / x[1]) if x[1] != 0. else np.pi / 2.
    invdistance = (1. / distance) if distance > 0. else 0.
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return 1.
    return float(distance)


@func_reg_deco
def deceptivemultimodal(x: np.ndarray) -> float:
    """Infinitely many local optima, as we get closer to the optimum."""
    assert len(x) >= 2
    distance = np.sqrt(x[0]**2 + x[1]**2)
    if distance == 0.:
        return 0.
    angle = np.arctan(x[0] / x[1]) if x[1] != 0. else np.pi / 2.
    invdistance = int(1. / distance) if distance > 0. else 0.
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return 1.
    return float(distance)


@func_reg_deco
def lunacek(x: np.ndarray) -> float:
    """Multimodal function.
    
    Based on https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/lunacek.html."""
    problemDimensions = len(x)
    s = 1.0 - (1.0 / (2.0 * np.sqrt(problemDimensions + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = - np.sqrt(abs((mu1**2 - 1.0) / s))
    firstSum = 0.0
    secondSum = 0.0
    thirdSum = 0.0
    for i in range(problemDimensions):
        firstSum += (x[i]-mu1)**2
        secondSum += (x[i]-mu2)**2
        thirdSum += 1.0 - np.cos(2*np.pi*(x[i]-mu1))
    return min(firstSum, 1.0*problemDimensions + secondSum)+10*thirdSum
