#!/usr/bin/env python3
"""Run the paper's inference simulations.

Compare methods on regular and nonregular factor-model populations,
including sensitivity to sample size and the near-optimality tolerance.
Tolerance comparisons reuse the same simulated samples.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('NUMBA_NUM_THREADS','3')
import argparse
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import inference_mc as mc
from mgp_core import RUN_OUTPUT_CACHE, atomic_write_pickle

BASE=dict(n=1000,zeta=1.,r=3,m=10,t=100,seed=20260901)
T_GRID=(50,100,250,1000,2500,10000)
CACHE=RUN_OUTPUT_CACHE/'mc_new_checkpoints'
CACHE_VERSION='paired-1000-v1'
REPLICATIONS=1000
BOOTSTRAP_DRAWS=199


def simulation_config(smoke=False):
    """Every method uses every replication of the same simulated panels."""
    reps=12 if smoke else REPLICATIONS
    return mc.McConfig(reps=reps,
        n_boot=39 if smoke else BOOTSTRAP_DRAWS,
        n_subsample=30 if smoke else 300,seed=0)


def design(name,label,**changes):
    return mc.Design(name=name,label=label,**{**BASE,**changes})


def selected_draw():
    """Return the first verified nonregular r=3 loading draw.

    The selection script redraws Gaussian loadings while keeping sizes,
    idiosyncratic variances, and the LPT partition fixed. The population is
    deliberately selected to be nonregular, not sampled to measure its prevalence.
    The saved audit records the selection and its certificates.
    """
    path=RUN_OUTPUT_CACHE/'selected_nonregular_zipf_r3.json'
    if not path.exists():
        from find_nonregular_zipf_population import main
        main()
    result=json.loads(path.read_text())
    assert result['seed']==BASE['seed'] and result['r']==BASE['r']
    assert result['audit']['regular'] is False
    return int(result['draw_index'])


def designs():
    redraw=selected_draw()
    main=[design('base','Baseline'),
          design('nonregular','Non-regular',loading_draw=redraw),
          design('r5',r'$r=5$',r=5)]
    main += [design(f'T{t}',fr'$T={t}$',t=t) for t in T_GRID if t!=100]
    main += [design('r1',r'$r=1$',r=1),design('r10',r'$r=10$',r=10)]
    main += [design(f'zeta{str(z).replace(".","")}',fr'$\zeta={z}$',zeta=z)
             for z in (.25,.75,1.25)]
    main += [design('M5',r'$M=5$',m=5),design('M30',r'$M=30$',m=30),
             design('ar30',r'AR(1), $\rho=0.3$',rho=.3),
             design('ar60',r'AR(1), $\rho=0.6$',rho=.6),
             design('t5',r'$t_5$ shocks',shock='t5'),
             design('homo','Homoskedastic',idio='homoskedastic')]
    sweep=[design(f'sweep_{tag}_T{t}',f'{tag}, T={t}',t=t,loading_draw=draw)
           for tag,draw in (('regular',1),('nonregular',redraw)) for t in T_GRID]
    return main,sweep


def is_comparison(d):
    return (d.r==3 and d.zeta==1 and d.m==10 and d.idio=='lognormal'
            and d.rho==0 and d.shock=='gaussian')


def regularity(d):
    if d.loading_draw>1:
        assert is_comparison(d)
        return False  # Verified by the saved pair of distinct optimal certificates.
    return mc.build_population(d).regular


def frozen_nonregular_population(d):
    """Load the selected draw and its verified population certificates.

    The planted share is feasible at the full-panel level, and the grouped
    dual certificate attains it. Therefore both population ceilings equal 3%.
    No additional full-panel SDP is needed to establish this equality.
    """
    arrays=np.load(RUN_OUTPUT_CACHE/'selected_nonregular_zipf_r3.npz')
    sizes=arrays['sizes'];g=arrays['grouping'];variance=arrays['sigma2_e']
    loading=g@arrays['loadings']
    sigma=loading@loading.T+np.diag((g*g)@variance)
    total=float(sigma.sum());gamma=float(np.sum(sizes*sizes*variance)/total)
    y=(arrays['dual_minus']+arrays['dual_plus'])/2
    assert np.linalg.eigvalsh(y).min()>-1e-9 and np.allclose(np.diag(y),1)
    assert np.max(np.abs(y@loading))<1e-8
    assert abs(float(np.sum(y*sigma))/total-gamma)<1e-8
    common=loading@loading.T
    a=y-gamma*np.ones((d.m,d.m))
    sigma_u=float(np.sqrt(2*np.trace(a@sigma@a@sigma))/total)
    return mc.Population(design=d,sizes=sizes,grouping=g,loads=(g*g).sum(axis=1),
        psi=variance,sigma_grouped=sigma,sigma_y2=total,gamma0=gamma,
        gamma_designed=gamma,gamma_bar0=gamma,ystar=y,common=common,
        tilde_rank=d.r,tilde_rank_tight=d.r,heywood=0,regular=False,
        sigma_u=sigma_u,half=mc._psd_half(sigma),grouped_loadings=loading)


def worker(payload):
    d,smoke=payload
    cfg=simulation_config(smoke)
    # Names/labels do not change the experiment. This also pairs tolerance runs.
    name=f'common_r{d.r}_z{d.zeta}_m{d.m}_idio{d.idio}_rho{d.rho}_shock{d.shock}'
    if d.loading_draw>1:name+=f'_draw{d.loading_draw}'
    numerical=replace(d,name=name,label='')
    fingerprint=hashlib.sha256(repr((numerical,cfg,CACHE_VERSION)).encode()).hexdigest()[:16]
    path=CACHE/f'{fingerprint}.pkl'
    if path.exists():
        result=pd.read_pickle(path)
        print(f'[inference_mc] reused {d.name}',flush=True)
    else:
        start=time.monotonic()
        families=('bootstrap','block_bootstrap') if d.rho else ('bootstrap',)
        if d.loading_draw>1:
            mc._POPULATION_CACHE[mc.population_key(numerical)]=frozen_nonregular_population(numerical)
        result=mc.run_design(numerical,cfg,resampling_families=families,
                            extra_delta_scales=(.5,2.) if is_comparison(d) else ())
        result.pop('population')
        result['row']['seconds']=time.monotonic()-start
        print(f"[inference_mc] {d.name}: {result['row']['seconds']:.1f}s; ceiling={result['row']['gamma_bar0']:.3%}",flush=True)
        atomic_write_pickle(result,path)
    result['row'].update(design=d.name,label=d.label,loading_draw=d.loading_draw,
                         regular=regularity(d))
    for row in result['coverage']:row.update(design=d.name,label=d.label)
    return d.name,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--only',default='')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,help='Optional separate progress bundle')
    args=parser.parse_args()
    main_designs,sweep=designs()
    # Run shared populations/T only once. Main baseline/T rows reuse the sweep.
    unique={}
    for d in [*sweep,*main_designs]:
        key=(mc.population_key(d),d.t,d.rho,d.shock)
        unique.setdefault(key,d)
    selected=list(unique.values())
    if args.only:selected=[d for d in selected if d.name in args.only.split(',')]
    # Population SDPs are one-time calculations. Complete them serially before
    # starting the sampling workers, which then read the same cached truths.
    prepared=set()
    for d in selected:
        key=mc.population_key(d)
        if key in prepared or d.loading_draw>1:
            continue
        mc.build_population(d)
        prepared.add(key)
    target=args.output or RUN_OUTPUT_CACHE/('mc_new_smoke_results.pkl' if args.smoke else 'mc_new_results.pkl')
    results={}
    def save():
        bundle={}
        for tag,ds in (('main',main_designs),('sweep',sweep)):
            bundle[tag]=[]
            for d in ds:
                key=(mc.population_key(d),d.t,d.rho,d.shock)
                if key not in results:continue
                import copy
                result=copy.deepcopy(results[key])
                result['row'].update(design=d.name,label=d.label)
                for row in result['coverage']:row.update(design=d.name,label=d.label)
                bundle[tag].append(result)
        bundle['delta']=[]
        for result in bundle['sweep']:
            for scale,coverage in result.get('tolerance_coverage',{}).items():
                row={**result['row'],'design':f"{result['row']['design']}_c{scale}",'delta_scale':scale}
                cov=[{**item,'design':row['design'],'label':row['label']} for item in coverage]
                bundle['delta'].append(dict(row=row,coverage=cov))
        bundle['meta']=dict(regular_r=3,nonregular_r=3,regular_draw=1,
                            nonregular_draw=selected_draw(),base=BASE,t_grid=T_GRID,
                            replications=simulation_config(args.smoke).reps,
                            bootstrap_draws=simulation_config(args.smoke).n_boot)
        atomic_write_pickle(bundle,target)
    if args.jobs==1:
        for d in selected:
            _,result=worker((d,args.smoke))
            results[(mc.population_key(d),d.t,d.rho,d.shock)]=result
            save()
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures={pool.submit(worker,(d,args.smoke)):d for d in selected}
            for future in as_completed(futures):
                d=futures[future];_,result=future.result()
                results[(mc.population_key(d),d.t,d.rho,d.shock)]=result
                save()
    save()
    print(target,flush=True)

if __name__=='__main__':main()
