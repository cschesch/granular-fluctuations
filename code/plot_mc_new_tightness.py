#!/usr/bin/env python3
"""Population tightness grid sharing the revised inference DGP.

N=1000; S_i proportional to i^-zeta; mean-one lognormal sigma_e,i^2;
Gaussian loadings independent of size. Each draw/rank's loading scale fixes
planted granularity at 3%. Gaussian aggregation is exact, so only M-dimensional
SDPs are required. No manuscript is edited.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from mgp_core import OUTPUT_DIR, RUN_OUTPUT_CACHE, lpt_buckets
from fast_mgp import solve_batch
from inference_mc import feasible_objectives, power_law_sizes
from mosek_face import licensed_mosek


def block(n, exponent, m, reps, gamma_target, idio_sd, seed, mu_end):
    sizes=power_law_sizes(n,exponent)
    buckets,loads=lpt_buckets(sizes**2,m)
    rng=np.random.default_rng([seed,int(round(1000*exponent)),m])
    sigma2_e=np.exp(idio_sd*rng.standard_normal((reps,n)))
    sigma2_e/=sigma2_e.mean(axis=1,keepdims=True)
    diagonal=np.array([(sigma2_e[:,bucket]*(sizes[bucket]**2)).sum(axis=1)
                       for bucket in buckets]).T
    planted=diagonal.sum(axis=1)
    z=rng.standard_normal((reps,m,m-1))
    gram=np.zeros((reps,m,m)); exposure=np.zeros(reps)
    eye=np.eye(m); root=np.sqrt(loads)
    rows=[]
    for r in range(1,m):
        col=z[:,:,r-1]
        gram+=col[:,:,None]*col[:,None,:]
        exposure+=(col@root)**2
        sigma2_lambda=planted*(1-gamma_target)/(gamma_target*exposure)
        sigma=sigma2_lambda[:,None,None]*gram+(diagonal/loads[None,:])[:,:,None]*eye
        sol=solve_batch(sigma,weights=loads,mu_end=mu_end,need_dual=False)
        values=np.asarray(sol.value).copy()
        gap=np.asarray(sol.relative_gap).copy()
        if r==1:
            # For one factor, the balancing inequality is an exact recovery
            # certificate. It also avoids subtracting huge nearly cancelling
            # covariances in draws with tiny aggregate factor exposure.
            absolute=np.abs(col*root[None,:])
            balanced=2*absolute.max(axis=1)<absolute.sum(axis=1)*(1-1e-12)
            values[balanced]=planted[balanced]
            gap[balanced]=0
        ok=gap<1e-6
        if not ok.all():
            scale=np.linalg.norm(sigma[~ok],axis=(1,2))*loads.mean()
            retry_mu=min(mu_end,float(np.min(np.abs(values[~ok])*1e-7/(2*m*scale))))
            retry=solve_batch(sigma[~ok],weights=loads,mu_end=retry_mu,need_dual=False,
                              max_iter=1200)
            values[~ok]=retry.value
            gap[~ok]=retry.relative_gap
            ok[~ok]=np.asarray(retry.relative_gap)<1e-6
        if not ok.all():
            # A few almost-cancelling factor draws need a conic solver rather
            # than the batched barrier. Check its primal/dual value agreement.
            import cvxpy as cp
            use_mosek = licensed_mosek()
            if use_mosek:
                options = dict(solver='MOSEK', mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-10})
            else:
                options = dict(solver='CLARABEL', tol_gap_abs=1e-12, tol_gap_rel=1e-11,
                               tol_feas=1e-12, max_iter=1000)
            for j in np.flatnonzero(~ok):
                covariance=sigma[j]*root[:,None]*root[None,:]
                scale=float(np.linalg.norm(covariance))
                covariance/=scale
                diagonal_var=cp.Variable(m,nonneg=True)
                y=cp.Variable((m,m),symmetric=True)
                primal=cp.Problem(cp.Maximize(cp.sum(diagonal_var)),
                                 [covariance-cp.diag(diagonal_var)>>0])
                dual=cp.Problem(cp.Minimize(cp.sum(cp.multiply(covariance,y))),
                               [y>>0,cp.diag(y)>=1])
                primal.solve(**options);dual.solve(**options)
                if use_mosek:
                    assert primal.status == dual.status == cp.OPTIMAL
                    lower, upper = primal.value, dual.value
                else:
                    solved = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                    if primal.status not in solved or dual.status not in solved:
                        raise RuntimeError(f"Tightness SDP statuses: {primal.status}, {dual.status}")
                    lower, upper = feasible_objectives(covariance, diagonal_var.value, y.value, np.ones(m), eps=2e-15)
                gap[j]=abs(upper-lower)/max(abs(upper),1e-30)
                assert gap[j]<1e-5,(exponent,m,r,gap[j])
                values[j]=upper*scale
        excess=np.maximum(values/planted-1,0)
        # The calibrated total variance is planted/gamma_target by construction.
        aggregate_loading=(z[:,:,:r]*root[None,:,None]).sum(axis=1)
        total=sigma2_lambda*np.sum(aggregate_loading**2,axis=1)+planted
        assert np.allclose(planted/total,gamma_target,rtol=1e-7,atol=1e-10)
        rows.append(dict(zeta=exponent,M=m,r=r,looseness=float(excess.mean()),
                         median_looseness=float(np.median(excess)),
                         recovery=float(np.mean(excess<=1e-6)),reps=reps,n=n,
                         idio_sd=idio_sd,gamma_target=gamma_target,seed=seed,
                         max_relative_gap=float(gap.max())))
    return rows


def r_max_exact(m: int, exponent: float) -> float:
    """(2/pi) kappa^2 (H_M^(zeta) - 1)^2, the threshold of Proposition ``prop:powerlaw``."""
    harmonic = np.sum(np.arange(1, m + 1, dtype=float) ** (-exponent))
    kappa = 1.0 + max(1.0 - exponent, 0.0) / np.sqrt(2.0 * exponent - 1.0)
    return (2.0 / np.pi) * kappa**2 * (harmonic - 1.0) ** 2


def render(table, stem) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import plot_style as ps

    ps.use()

    exponents = sorted(table["zeta"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), sharex=True, sharey=True)
    clip = float(np.quantile(table["looseness"], 0.95))
    # A display floor keeps zero (and tiny negative solver error) visible on a
    # logarithmic scale. It does not change the cached simulation results.
    floor = 1e-4
    norm = LogNorm(vmin=floor, vmax=max(clip, 10 * floor), clip=True)
    mesh = None
    handles: dict = {}
    for ax, exponent in zip(axes.ravel(), exponents):
        block = table[table["zeta"].eq(exponent)]
        grid = block.pivot(index="r", columns="M", values="looseness")
        mesh = ax.pcolormesh(grid.columns.values, grid.index.values,
                             np.maximum(grid.values, floor),
                             cmap="viridis", norm=norm, shading="nearest")
        ms = grid.columns.values.astype(float)
        ax.grid(False)
        if exponent < 0.5:
            line, = ax.plot(ms, ms - 2.0 * np.sqrt(ms), color="tab:red",
                            lw=2.0, zorder=5)
            handles.setdefault(r"Balanced threshold $r=M-2\sqrt{M}$", line)
        elif exponent > 0.5:
            line, = ax.plot(ms, [r_max_exact(int(v), exponent) for v in ms],
                            color="tab:orange", lw=2.0, zorder=5)
            handles.setdefault(r"Power-law threshold $r_{\max}(M,\zeta)$", line)
        ax.set_title(rf"$\zeta={exponent:g}$")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_xticks(np.arange(0, 51, 10))
        ax.set_yticks(np.arange(0, 51, 10))
    for ax in axes[1]:
        ax.set_xlabel(r"Number of buckets $M$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Number of factors $r$")
    fig.legend(handles.values(), handles.keys(), fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.45, -0.02))
    ticks=[10.0**k for k in range(-4, int(np.floor(np.log10(norm.vmax)))+1)]
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="max",
                           ticks=ticks,
                           label="Mean grouped upper bound looseness "
                           r"$(\tilde\gamma-\gamma)/\gamma$")
    colorbar.set_ticklabels([r"$\leq 10^{-4}$"]+[rf"$10^{{{int(np.log10(v))}}}$" for v in ticks[1:]])
    path = (OUTPUT_DIR / stem).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[tightness] wrote {path}", flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reps',type=int,default=100)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--n',type=int,default=1000)
    p.add_argument('--m-min',type=int,default=2)
    p.add_argument('--m-max',type=int,default=50)
    p.add_argument('--zetas',default='0.25,0.75,1,1.25')
    p.add_argument('--gamma-target',type=float,default=.03)
    p.add_argument('--idio-sd',type=float,default=.5)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--mu-end',type=float,default=1e-9)
    p.add_argument('--stem',default='mc_new_tightness')
    p.add_argument('--from-cache',action='store_true')
    p.add_argument('--restart',action='store_true')
    a=p.parse_args()
    path=RUN_OUTPUT_CACHE/f'{a.stem}.csv'
    if a.from_cache:
        table=pd.read_csv(path)
    else:
        old=pd.read_csv(path) if path.exists() and not a.restart else pd.DataFrame()
        if not old.empty:
            assert (old.reps==a.reps).all() and (old.n==a.n).all()
            assert (old.idio_sd==a.idio_sd).all() and (old.gamma_target==a.gamma_target).all()
            assert (old.seed==a.seed).all()
        rows=old.to_dict('records'); start=time.monotonic()
        pending=[]
        for exponent in map(float,a.zetas.split(',')):
            for m in range(a.m_min,a.m_max+1):
                previous=[row for row in rows if row['zeta']==exponent and row['M']==m]
                if len(previous)==m-1:
                    continue
                assert not previous, 'Incomplete group-count checkpoint'
                pending.append((exponent,m))
        # Different group counts are independent. Only the parent writes caches.
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futures={pool.submit(block,a.n,z,m,a.reps,a.gamma_target,a.idio_sd,a.seed,a.mu_end):(z,m)
                     for z,m in sorted(pending,key=lambda x:-x[1])}
            for future in as_completed(futures):
                exponent,m=futures[future]
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(path,index=False)
                print(f'zeta={exponent:g}, M={m}; {time.monotonic()-start:.1f}s',flush=True)
        table=pd.DataFrame(rows)
    render(table, a.stem)
    print(table.groupby('zeta').agg(cells=('r','size'),max_excess=('looseness','max'),
                                   max_gap=('max_relative_gap','max')).to_string(),flush=True)

if __name__=='__main__':
    main()
