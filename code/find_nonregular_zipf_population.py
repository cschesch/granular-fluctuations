#!/usr/bin/env python3
"""Select the first strictly certified nonregular r=3 Zipf loading draw.

Sizes, LPT partition and idiosyncratic variances are fixed. Only raw Gaussian
loadings are redrawn, sequentially from the existing population RNG. Selection
is explicit; this is not an estimate of the prevalence of nonregularity.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from scipy.linalg import null_space
from mgp_core import RUN_OUTPUT_CACHE,lpt_buckets
from check_zipf_regularity import primal
from mc_regularity_audit import audit_covariance
from mosek_face import licensed_mosek
import cvxpy as cp


def main():
    n,m,r,seed=1000,10,3,20260901
    rng=np.random.default_rng(seed)
    s=1/np.arange(1,n+1,dtype=float);s/=s.sum()
    variance=np.exp(.5*rng.standard_normal(n));variance/=variance.mean()
    groups,_=lpt_buckets(s*s,m)
    g=np.zeros((m,n))
    for j,group in enumerate(groups):g[j,group]=s[group]
    granular=float(np.sum(s*s*variance))
    attempts=[]
    for draw in range(1,1001):
        raw=rng.standard_normal((n,r))
        raw_group=g@raw
        q=null_space(raw_group.T,rcond=1e-12)
        d=q.shape[1]
        k=cp.Variable((d,d),symmetric=True)
        floor=cp.Variable()
        problem=cp.Problem(cp.Maximize(floor),[k-floor*np.eye(d)>>0,cp.diag(q@k@q.T)==1])
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10,max_iter=1000)
        assert problem.status in ('optimal','optimal_inaccurate'),problem.status
        margin=float(floor.value)
        attempts.append(dict(draw=draw,certificate_floor=margin,status=problem.status))
        print(f'draw {draw}: certificate floor {margin:.8f}',flush=True)
        if margin<1e-4:continue
        theta=granular*.97/(.03*np.sum((s@raw)**2))
        loading=np.sqrt(theta)*raw_group
        sigma=loading@loading.T+np.diag((g*g)@variance)
        aggregate=float(sigma.sum());sigma/=aggregate;loading/=np.sqrt(aggregate)
        # An explicit nonzero null perturbation of the diagonal equations,
        # using an orthonormal basis of symmetric matrices.
        basis=[]
        for a in range(d):
            for b in range(a,d):
                e=np.zeros((d,d));e[a,b]=e[b,a]=1 if a==b else 1/np.sqrt(2)
                basis.append(e)
        linear=np.column_stack([np.diag(q@e@q.T) for e in basis])
        kernel=null_space(linear,rcond=1e-10)
        assert kernel.shape[1]>0
        perturb=sum(coef*e for coef,e in zip(kernel[:,0],basis))
        perturb/=np.linalg.norm(perturb,2)
        k0=np.asarray(k.value)
        step=.25*np.linalg.eigvalsh(k0).min()
        certs=[q@(k0+sign*step*perturb)@q.T for sign in (-1,1)]
        for cert in certs:
            assert np.linalg.eigvalsh(cert).min()>-1e-9
            assert np.max(np.abs(np.diag(cert)-1))<1e-8
            assert np.max(np.abs(cert@loading))<1e-8
            assert abs(np.sum(cert*sigma)-.03)<1e-8
        separation=float(np.linalg.norm(certs[0]-certs[1]))
        assert separation>1e-3
        clarabel,_,_=primal(sigma,'CLARABEL')
        assert abs(clarabel-.03)<1e-8
        mosek = None
        if licensed_mosek():
            mosek,_,_=primal(sigma,'MOSEK')
            assert abs(mosek-.03)<1e-8
        audit=audit_covariance(sigma)
        assert audit['regular'] is False,audit
        info=dict(seed=seed,draw_index=draw,attempts=attempts,N=n,M=m,r=r,zeta=1,
                  idiosyncratic_variances='fixed original lognormal draw, mean one',
                  loading_draws='successive N by 3 standard normal arrays after fixed idiosyncratic draw',
                  sigma_lambda2=float(theta),gamma_designed=.03,grouped_ceiling_clarabel=clarabel,
                  grouped_ceiling_mosek=mosek,certificate_floor=margin,
                  symmetric_unknowns=d*(d+1)//2,diagonal_equations=m,
                  diagonal_map_rank=int(np.linalg.matrix_rank(linear,tol=1e-10)),
                  null_directions=int(kernel.shape[1]),two_certificate_separation=separation,
                  audit=audit)
        stem=RUN_OUTPUT_CACHE/'selected_nonregular_zipf_r3'
        stem.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        np.savez_compressed(stem.with_suffix('.npz'),sizes=s,grouping=g,sigma2_e=variance,
                            raw_loadings=raw,loadings=np.sqrt(theta)*raw,
                            sigma_grouped_normalized=sigma,dual_minus=certs[0],dual_plus=certs[1])
        print(json.dumps(info,indent=2),flush=True)
        return
    raise RuntimeError('No certified nonregular population in first 1000 loading draws')

if __name__=='__main__':main()
