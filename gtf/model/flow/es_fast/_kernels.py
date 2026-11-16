"""Triton kernels of the nearest-support search: FP32 screening, then FP64 refinement of ambiguous queries."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["NQ", "NS"])
def _screen_kernel(Q, S, DIST, INDEX, AMBIG, NQ, NS,
                   BQ: tl.constexpr, BS: tl.constexpr):
    iq = tl.program_id(0) * BQ + tl.arange(0, BQ)
    js = tl.arange(0, BS)
    qx = tl.load(Q + 3 * iq, iq < NQ, 0)
    qy = tl.load(Q + 3 * iq + 1, iq < NQ, 0)
    qz = tl.load(Q + 3 * iq + 2, iq < NQ, 0)
    best = tl.full((BQ,), float("inf"), tl.float32)
    second = tl.full((BQ,), float("inf"), tl.float32)
    best_i = tl.full((BQ,), 2147483647, tl.int32)
    for start in range(tl.cdiv(NS, BS)):
        idx = start * BS + js
        sx = tl.load(S + 3 * idx, idx < NS, 0)
        sy = tl.load(S + 3 * idx + 1, idx < NS, 0)
        sz = tl.load(S + 3 * idx + 2, idx < NS, 0)
        dx = qx[:, None] - sx[None, :]
        dy = qy[:, None] - sy[None, :]
        dz = qz[:, None] - sz[None, :]
        square = (dx * dx + dy * dy) + dz * dz
        square = tl.where((iq[:, None] < NQ) & (idx[None, :] < NS),
                          square, float("inf"))
        value = tl.min(square, 1)
        index = tl.min(tl.where((square == value[:, None]) & (idx[None, :] < NS),
                                idx[None, :], 2147483647), 1)
        runner = tl.min(tl.where(idx[None, :] != index[:, None], square,
                                 float("inf")), 1)
        improve = (value < best) | ((value == best) & (index < best_i))
        second = tl.where(improve, tl.minimum(best, runner),
                          tl.minimum(second, value))
        best = tl.minimum(best, value)
        best_i = tl.where(improve, index, best_i)

    # Direct 3D squares use 3 subtracts, 3 multiplies and 2 adds. With no FMA,
    # <= 8 rounded FP32 operations, gamma_8 < 4.77e-7 (u = 2^-24). The wider
    # 2^-18 relative enclosure allows interval-construction rounding too.
    # 2^-119 absolute slack exceeds normal/FTZ underflow loss at every operation.
    # Positive overflow cannot look artificially small: an infinite best (all
    # overflow) is always refined. NS=1 needs no ambiguity comparison.
    upper = best * (1.0 + 0.000003814697265625) + 1.504632769052528e-36
    lower_other = second * (1.0 - 0.000003814697265625) - 1.504632769052528e-36
    ambiguous = (best == float("inf")) | ((NS > 1) & (lower_other <= upper))
    tl.store(DIST + iq, tl.sqrt(best), iq < NQ)
    tl.store(INDEX + iq, best_i.to(tl.int64), iq < NQ)
    tl.store(AMBIG + iq, ambiguous, iq < NQ)


@triton.jit(do_not_specialize=["NQ", "NS"])
def _refine_kernel(Q, S, DIST, INDEX, AMBIG, NQ, NS,
                   BQ: tl.constexpr, BS: tl.constexpr):
    iq = tl.program_id(0) * BQ + tl.arange(0, BQ)
    ambiguous = tl.load(AMBIG + iq, iq < NQ, 0).to(tl.int32)
    chosen_i = tl.load(INDEX + iq, iq < NQ, 0)
    if tl.sum(ambiguous, 0) > 0:
        active = (iq < NQ) & (ambiguous != 0)
        js = tl.arange(0, BS)
        qx = tl.load(Q + 3 * iq, active, 0).to(tl.float64)
        qy = tl.load(Q + 3 * iq + 1, active, 0).to(tl.float64)
        qz = tl.load(Q + 3 * iq + 2, active, 0).to(tl.float64)
        best = tl.full((BQ,), float("inf"), tl.float64)
        best_i = tl.full((BQ,), 2147483647, tl.int32)
        for start in range(tl.cdiv(NS, BS)):
            idx = start * BS + js
            sx = tl.load(S + 3 * idx, idx < NS, 0).to(tl.float64)
            sy = tl.load(S + 3 * idx + 1, idx < NS, 0).to(tl.float64)
            sz = tl.load(S + 3 * idx + 2, idx < NS, 0).to(tl.float64)
            dx = qx[:, None] - sx[None, :]
            dy = qy[:, None] - sy[None, :]
            dz = qz[:, None] - sz[None, :]
            square = (dx * dx + dy * dy) + dz * dz
            square = tl.where(active[:, None] & (idx[None, :] < NS),
                              square, float("inf"))
            value = tl.min(square, 1)
            index = tl.min(tl.where((square == value[:, None]) & (idx[None, :] < NS),
                                    idx[None, :], 2147483647), 1)
            improve = (value < best) | ((value == best) & (index < best_i))
            best = tl.minimum(value, best)
            best_i = tl.where(improve, index, best_i)
        chosen_i = tl.where(active, best_i.to(tl.int64), chosen_i)
    # The screen certifies the selected index, not the FP32 distance rounding: one
    # FP64 3D distance per query keeps screening rounding (and underflow) out of the
    # returned distance.
    qx = tl.load(Q + 3 * iq, iq < NQ, 0).to(tl.float64)
    qy = tl.load(Q + 3 * iq + 1, iq < NQ, 0).to(tl.float64)
    qz = tl.load(Q + 3 * iq + 2, iq < NQ, 0).to(tl.float64)
    sx = tl.load(S + 3 * chosen_i, iq < NQ, 0).to(tl.float64)
    sy = tl.load(S + 3 * chosen_i + 1, iq < NQ, 0).to(tl.float64)
    sz = tl.load(S + 3 * chosen_i + 2, iq < NQ, 0).to(tl.float64)
    dx, dy, dz = qx - sx, qy - sy, qz - sz
    square = (dx * dx + dy * dy) + dz * dz
    tl.store(DIST + iq, tl.sqrt(square).to(tl.float32), iq < NQ)
    tl.store(INDEX + iq, chosen_i, iq < NQ)
