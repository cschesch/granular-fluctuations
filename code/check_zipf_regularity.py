#!/usr/bin/env python3
"""Focused fixed-population Zipf audit: fresh versus nested factor draws.

No sampling Monte Carlo and no manuscript edits. Cross-checks two conic solvers,
primal complementary geometry, and dual solution ranges at shrinking gaps.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import argparse
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from mgp_core import RUN_OUTPUT_CACHE, OUTPUT_DIR, lpt_buckets
from mc_regularity_audit import audit_covariance
import cvxpy as cp


def population(r,seed=20260901,nested=False):
    n,m=1000,10
    rng=np.random.default_rng(seed)
    s=1/np.arange(1,n+1,dtype=float);s/=s.sum()
    variance=np.exp(.5*rng.standard_normal(n));variance/=variance.mean()
    # Same matrix call as the existing code, or a common ten-column matrix.
    raw=rng.standard_normal((n,10 if nested else r))[:,:r]
    groups,_=lpt_buckets(s*s,m)
    g=np.zeros((m,n))
    for k,group in enumerate(groups):g[k,group]=s[group]
    theta=float(np.sum(s*s*variance)*.97/(.03*np.sum((s@raw)**2)))
    loading=np.sqrt(theta)*(g@raw)
    diag=(g*g)@variance
    sigma=loading@loading.T+np.diag(diag)
    scale=float(sigma.sum())
    return sigma/scale,loading/np.sqrt(scale),diag/scale,theta


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


def planted_certificate(loading):
    """Maximize minimum eigenvalue on the exact planted dual face.

    Y=QKQ', diag(Y)=1 and K positive definite certifies a genuine nontrivial
    dual face whenever symmetric K has more free entries than diagonal equations.
    """
    m,r=loading.shape
    q=null_space(loading.T,rcond=1e-12)
    k=cp.Variable((m-r,m-r),symmetric=True)
    floor=cp.Variable()
    problem=cp.Problem(cp.Maximize(floor),[k-floor*np.eye(m-r)>>0,
                                            cp.diag(q@k@q.T)==1])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10,max_iter=1000)
    if problem.status in ('optimal','optimal_inaccurate'):
        residual=float(np.max(np.abs(np.diag(q@k.value@q.T)-1)))
        return float(floor.value),residual
    return None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20260901)
    args=parser.parse_args()
    rows=[];details=[]
    for nested in (False,True):
        for r in (1,2,3,5,10):
            sigma,loading,diag,theta=population(r,args.seed,nested)
            audit=audit_covariance(sigma,probes=3,cap_tolerances=(1e-5,1e-7,1e-9))
            cv,d,status=primal(sigma,'CLARABEL')
            try:
                mv,md,mstatus=primal(sigma,'MOSEK')
                assert abs(cv-mv)<1e-7,(cv,mv)
                mosek_error=None
            except cp.error.SolverError as err:
                mv=None;mstatus='unavailable';mosek_error=str(err).splitlines()[0]
            if r<10:
                floor,resid=planted_certificate(loading)
            else:floor=resid=None
            ratio=float(np.max(np.abs(loading[:,0]))/(np.abs(loading[:,0]).sum()-np.max(np.abs(loading[:,0])))) if r==1 else None
            record=dict(draw='nested' if nested else 'existing',r=r,
                        grouped_ceiling=cv,planted_share=float(diag.sum()),
                        regular=audit['regular'],common_rank=10-audit['nullity'],
                        positive_optimal_diagonals=int(np.sum(d>1e-7)),
                        certificate_floor=floor,certificate_residual=resid,
                        width_1e5=max(audit['probe_widths'][0]),
                        width_1e7=max(audit['probe_widths'][1]),
                        width_1e9=max(audit['probe_widths'][2]),
                        mosek_ceiling=mv,mosek_status=mstatus,
                        one_factor_imbalance_ratio=ratio,sigma_lambda2=theta)
            rows.append(record)
            details.append(dict(**record,audit=audit,mosek_error=mosek_error))
            print(json.dumps(record),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR/'zipf_regularity_audit.csv',index=False)
    (RUN_OUTPUT_CACHE/'zipf_regularity_audit.json').write_text(json.dumps(details,indent=2)+'\n')
    print(frame.to_string(index=False),flush=True)

if __name__=='__main__':main()
