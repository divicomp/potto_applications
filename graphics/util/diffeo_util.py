from dataclasses import dataclass
from typing import Type
import itertools

from potto import (
    Int,
    Const,
    Measure,
    Sym,
    Var,
    GExpr,
    TegVar,
    Heaviside,
    Diffeomorphism,
    Mul,
    Add,
    Div,
    Abs,
    Sample,
    VarVal,
    Samples,
)
from potto import simplify
from potto.lang.evaluate import evaluate


def lookup_tvar_in_expr(expr: GExpr, tvar: GExpr) -> bool:
    match expr:
        case Div(l, r) | Mul(l, r) | Add(l, r):
            return lookup_tvar_in_expr(l, tvar) or lookup_tvar_in_expr(r, tvar)
        case TegVar():
            return expr == tvar
        case Var() | Const():
            return False
        case _:
            raise TypeError(f"Could not match on type {type(expr)}")


@dataclass(frozen=True)
class LinearDiffeo(Diffeomorphism):
    diffeo_expr: GExpr = Const(0)

    def __str__(self):
        return f"linear_diffeo({[str(i) for i in self.vars]}, {[str(i) for i in self.tvars]})"

    def function(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
        d = {v.name.uid: var for v, var in zip(self.vars + self.tvars, vars + tvars)}
        f = self.holes(self.diffeo_expr, d)  # TODO: simplify diffeo expression
        f = simplify(f)
        return f, tvars[1]

    def inverse(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
        inv_expr = self.invert(self.diffeo_expr, self.tvars[0], tvars[0])
        d = {v.name.uid: var for v, var in zip(self.vars + self.tvars, vars + tvars)}
        finv = self.holes(inv_expr, d)
        finv = simplify(finv)

        return finv, tvars[1]

    def bounds_transfer(self, lower_left_corner: tuple, upper_right_corner: tuple, env):
        corners = itertools.product(*zip(lower_left_corner, upper_right_corner))
        corner_images = []
        for corner in corners:
            funs = self.function(self.vars, corner)
            corner_images.append([evaluate(f, env) for f in funs])
        mins = tuple(min(xs) for xs in zip(*corner_images))
        maxs = tuple(max(xs) for xs in zip(*corner_images))
        return mins, maxs

    @classmethod
    def holes(cls, e: GExpr, to_replace: dict[int, GExpr]) -> GExpr:
        """Formally substitutes the variables in to_replace into e."""
        match e:
            case Div(l, r) | Mul(l, r) | Add(l, r):
                return type(e)(cls.holes(l, to_replace), cls.holes(r, to_replace))
            case TegVar() | Var():
                if e.name.uid not in to_replace:
                    return e
                return to_replace[e.name.uid]
            case Const():
                return e
            case _:
                raise TypeError(f"Could not match on type {type(e)}")

    @classmethod
    def coefficient_of(cls, e: GExpr, v: TegVar) -> GExpr:  # TODO (jesse): better
        return cls._coefficient_of(e, v) - cls._coefficient_of(e, None)

    @classmethod
    def _coefficient_of(cls, e: GExpr, v: TegVar) -> GExpr:
        """Remove the Var or TegVar v from the (affine) expression e."""
        match e:
            case Div(l, r) | Mul(l, r) | Add(l, r):
                return type(e)(cls._coefficient_of(l, v), cls._coefficient_of(r, v))
            case TegVar():
                return Const(1) if v == e else Const(0)
            case Var() | Const():
                return e
            case _:
                raise TypeError(f"Could not match on type {type(e)}")

    @classmethod
    def remove(cls, e: GExpr, v: TegVar) -> GExpr:
        """ "Remove the Var or TegVar v from the (affine) expression e."""
        match e:
            case Div(l, r) | Mul(l, r) | Add(l, r):
                return type(e)(cls.remove(l, v), cls.remove(r, v))
            case TegVar() | Var():
                return Const(0) if v == e else e
            case Const():
                return e
            case _:
                raise TypeError(f"Could not match on type {type(e)}")

    # z1 = 1 + 2t + 3x + 4y
    # z2 = y
    # x = (1 + 2t + 4z2 - z1) / -3
    @classmethod
    def invert(cls, e, v, z1):
        if not lookup_tvar_in_expr(e, v):
            raise ValueError(f"Could not find tvar {v} in {e}")
        c = cls.coefficient_of(e, v)
        e = cls.remove(e, v)
        return (z1 - e) / c


def make_diffeo(expr: GExpr, vs: tuple[GExpr, ...], tvs: tuple[GExpr, ...]) -> Diffeomorphism:
    """Turn an expression that is a function of a variable and two variables of integration into a diffeomorphism."""

    def closure(expr: GExpr, vs: tuple[Var], tvs: tuple[TegVar]):
        new_tvs = []
        for tv in tvs:
            if lookup_tvar_in_expr(expr, tv):
                new_tvs.append(tv)
        new_tvs = tuple(new_tvs)
        tvs = new_tvs
        if len(tvs) == 0:
            raise TypeError(f"No variables of integration in expr {expr}")
        # 1 / |dw(x) / dx| (x_tilde)
        weight = 1 / Abs(LinearDiffeo.coefficient_of(expr, tvs[0]))
        return LinearDiffeo(vs, tvs, weight, diffeo_expr=expr)

    return closure(expr, vs, tvs)
