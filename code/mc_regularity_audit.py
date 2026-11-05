#!/usr/bin/env python3
"""Direct numerical singleton audit of the grouped-MGP population dual.

Rank at a coarse threshold is not a singleton test, especially at Heywood
boundaries. This checks the diagonal linear system in a high-accuracy primal
kernel and ranges of several deterministic linear dual objectives under
shrinking optimality caps. It never changes the estimator or bootstrap.
"""
from __future__ import annotations
import numpy as np
import cvxpy as cp

OPTIONS=dict(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10,max_iter=1000)


def primal(sigma,solver):
    m=len(sigma)
    d=cp.Variable(m,nonneg=True)
    problem=cp.Problem(cp.Maximize(cp.sum(d)),[sigma-cp.diag(d)>>0])
    if solver=='CLARABEL':
        options=dict(tol_gap_abs=1e-11,tol_feas=1e-11,tol_gap_rel=1e-11,max_iter=1000)
    else:
        options=dict(mosek_params={'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-10,
                                  'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-10,
                                  'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-10})
    problem.solve(solver=solver,**options)
    assert problem.status in ('optimal','optimal_inaccurate'),problem.status
    return float(problem.value),np.asarray(d.value),problem.status


def audit_covariance(sigma,probes=3,cap_tolerances=(1e-5,1e-7,1e-9)):
    sigma=np.asarray(sigma,dtype=float);sigma=(sigma+sigma.T)/2
    sigma=sigma/float(sigma.sum())
    m=len(sigma)
    diag=cp.Variable(m,nonneg=True)
    primal=cp.Problem(cp.Maximize(cp.sum(diag)),[sigma-cp.diag(diag)>>0])
    primal.solve(**OPTIONS)
    assert primal.status in ('optimal','optimal_inaccurate'),primal.status
    y=cp.Variable((m,m),symmetric=True)
    constraints=[y>>0,cp.diag(y)>=1]
    dual=cp.Problem(cp.Minimize(cp.sum(cp.multiply(sigma,y))),constraints)
    dual.solve(**OPTIONS)
    assert dual.status in ('optimal','optimal_inaccurate'),dual.status
    optimum=max(float(primal.value),float(dual.value))
    common=sigma-np.diag(diag.value)
    eig,u=np.linalg.eigh(common)
    kernel=u[:,eig<1e-7]
    q=kernel.shape[1]
    active=np.asarray(diag.value)>1e-7
    linear=np.array([[kernel[i,a]*kernel[i,b]*(1 if a==b else 2)
                      for a in range(q) for b in range(a,q)] for i in np.flatnonzero(active)])
    rank=int(np.linalg.matrix_rank(linear,tol=1e-7)) if linear.size else 0
    columns=q*(q+1)//2
    cap=cp.Parameter(nonneg=True)
    h=cp.Parameter((m,m),symmetric=True)
    program=cp.Problem(cp.Maximize(cp.sum(cp.multiply(h,y))),constraints+[cp.sum(cp.multiply(sigma,y))<=cap])
    rng=np.random.default_rng(6)
    matrices=[]
    for _ in range(probes):
        h0=rng.normal(size=(m,m));h0=(h0+h0.T)/2;h0/=np.linalg.norm(h0)
        matrices.append(h0)
    widths=[]
    statuses=[]
    for tol in cap_tolerances:
        cap.value=optimum+tol
        this=[]
        for h0 in matrices:
            endpoints=[]
            for sign in (-1,1):
                h.value=sign*h0
                program.solve(**OPTIONS)
                assert program.status in ('optimal','optimal_inaccurate'),program.status
                statuses.append(program.status)
                endpoints.append(float(program.value))
            this.append(sum(endpoints))
        widths.append(this)
    widths=np.asarray(widths)
    contraction=float(np.max(widths[-1]/np.maximum(widths[-2],1e-12)))
    # At a singleton, widths normally contract at sqrt(cap tolerance).
    # A nontrivial face retains a positive width. Report unresolved if neither.
    if rank==columns or contraction<.35:
        regular=True
    elif contraction>.7 and widths[-1].max()>.01:
        regular=False
    else:
        regular=None
    return dict(regular=regular,primal_value=float(primal.value),dual_value=float(dual.value),
                primal_dual_gap=float(dual.value-primal.value),
                common_eigenvalues=eig.tolist(),active_diagonal_count=int(active.sum()),
                nullity=q,diagonal_system_columns=columns,diagonal_system_rank=rank,
                diagonal_system_injective=rank==columns,cap_tolerances=list(cap_tolerances),
                probe_widths=widths.tolist(),last_width_contraction=contraction,
                inaccurate_solves=statuses.count('optimal_inaccurate'))
