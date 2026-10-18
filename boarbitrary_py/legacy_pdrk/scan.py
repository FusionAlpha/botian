from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .config import LegacyContext, LegacyScanPoint


def iter_legacy_scan_points(ctx: LegacyContext) -> Iterator[LegacyScanPoint]:
    scan = ctx.case.scan
    par_base = np.zeros(6, dtype=float)
    par_base[0] = scan.initial_k
    par_base[1] = scan.initial_theta_deg
    par_base[2] = np.cos(np.deg2rad(par_base[1])) * par_base[0]
    par_base[3] = np.sin(np.deg2rad(par_base[1])) * par_base[0]
    flat_index = 0
    for jpa, pa_value in enumerate(ctx.pa):
        for jpb, pb_value in enumerate(ctx.pb):
            par = par_base.copy()
            par[scan.ipa - 1] = pa_value if scan.iloga == 0 else 10.0**pa_value
            if scan.ipa != scan.ipb:
                par[scan.ipb - 1] = pb_value if scan.ilogb == 0 else 10.0**pb_value

            k = abs(par[0]) * ctx.kn
            theta_deg = float(par[1])
            if scan.ipa > 2 and scan.ipb > 2:
                kz = par[2] * ctx.kn
                kx = par[3] * ctx.kn
            else:
                kz = np.cos(np.deg2rad(theta_deg)) * k
                kx = np.sin(np.deg2rad(theta_deg)) * k
            k = float(np.hypot(kx, kz))
            theta_deg = float(np.rad2deg(np.angle(kz + 1j * kx)))
            yield LegacyScanPoint(
                flat_index=flat_index,
                jpa=jpa,
                jpb=jpb,
                pa_value=float(pa_value),
                pb_value=float(pb_value),
                par=par,
                k_norm=k / ctx.kn,
                theta_deg=theta_deg,
                kz_norm=kz / ctx.kn,
                kx_norm=kx / ctx.kn,
                k=k,
                kz=float(kz),
                kx=float(kx),
            )
            flat_index += 1
