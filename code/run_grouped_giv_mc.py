#!/usr/bin/env python3
"""Run the paper's grouped-GIV Monte Carlo, resuming the expensive long cell.

One full-panel eigendecomposition is shared by PCA and both Hahn rows.
Symmetric covariance cross-products
and contraction before the naive-variance cross-product save computation and
avoid duplicate T x K influence arrays; the
unreported covariance condition number is skipped. Checkpoints retain the exact
RNG state after each batch.
"""
from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
import scipy.linalg as la
import grouped_giv_mc as mc
from grouped_giv import (InstrumentSpace,estimated_instrument_space,estimate_joint_giv,
                         hahn_bic,projected_weight,scalar_iv,_failed_equation,
                         joint_covariance_failure)
from mgp_core import OUTPUT_DIR, RUN_OUTPUT_CACHE, atomic_write_pickle

CELLS = mc.SAMPLES
PREFIX='grouped_giv_mc_fixed_n'


def full_spectrum(x):
    """The N x N covariance after removing the known price direction."""
    t,n=x.shape
    residual=x-x.mean(axis=1,keepdims=True)
    covariance=la.blas.dsyrk(1./t,residual.T,lower=1)
    values,vectors=la.eigh(covariance,lower=True)
    # The first eigenvector is the known, zero-eigenvalue price direction.
    return values[1:],vectors[:,1:]


def space_from_spectrum(values,vectors,rank,d,bic):
    n=vectors.shape[0];k=n-rank
    basis=vectors[:,:k]
    if rank>1:
        factors=vectors[:,k:]
        inverse=1./(values[0]-values[k:])
        upsilon=(factors*inverse)@factors.T
        gap=float(values[k]-values[0])
    else:
        upsilon=np.zeros((n,n));gap=float('nan')
    return InstrumentSpace(basis=basis,combined_rank=rank,residual_dimension=k,
                           information=float(np.sum((basis.T@d)**2)),eigvals=values,
                           noise_variance=float(values[0]),eigengap=gap,upsilon=upsilon,
                           bic_values=bic,estimated=True)


def simulate_cell(pop,reps,rng,offset=0):
    n,t=pop.n,pop.t
    rows=[]
    ones=np.ones(n);loading=pop.loadings[:,0]
    b,d=pop.price_direction,pop.aggregate_direction
    aggregate_loading=float(pop.sizes@loading)
    for replication in range(offset,offset+reps):
        factor=rng.standard_normal(t)
        innovation=rng.standard_normal(t)
        epsilon=mc.SIGMA_EPS*(mc.FACTOR_EPS_CORR*factor+np.sqrt(1-mc.FACTOR_EPS_CORR**2)*innovation)
        error=rng.standard_normal((t,n))*mc.SIGMA_E
        aggregate=(mc.PHI*epsilon+aggregate_loading*factor+error@pop.sizes)/(1-mc.PHI*mc.PSI)
        price=mc.PSI*aggregate+epsilon
        x=error
        x+=np.outer(price,mc.PHI*ones)
        x+=np.outer(factor,loading)
        grouped=x@pop.whitening.T
        values,vectors=full_spectrum(x)
        rows.extend(mc.bai_ng_records(pop,replication,x,grouped,price,aggregate,(values,vectors)))
        pca_loading=vectors[:,-1:]
        weight=mc._project_off(pop.sizes,np.column_stack((ones,pca_loading)))
        phi,psi=mc._full_panel(pop,x,weight,price,aggregate)
        def record(method,phi,psi,rank,information,**kw):
            rows.append(mc._record(pop,replication,method,phi,psi,selected_rank=rank,
                                   information=information,**kw))
        record('full_pca',phi,psi,2,float(weight@weight),
               pca_overlap=float((loading@pca_loading[:,0])**2/(loading@loading)),
               factor_leak=float(weight@loading/(pop.full_baseline_weight@loading)))
        rank,bic=hahn_bic(values,t)
        for method,chosen in [('full_rank_known',2),('full_hahn',rank)]:
            reason=joint_covariance_failure(t,n-chosen)
            if reason:
                phi=psi=_failed_equation(reason)
                information=float('nan');gap=float('nan')
            else:
                space=space_from_spectrum(values,vectors,chosen,pop.sizes,bic)
                result=estimate_joint_giv(x,price,aggregate,ones,pop.sizes,space,diagnostics=False)
                phi,psi=result.phi,result.psi
                information=space.information;gap=space.eigengap
            record(method,phi,psi,chosen,information,eigengap=gap)
        phi,psi=mc._full_panel(pop,x,pop.full_oracle_weight,price,aggregate)
        record('full_oracle',phi,psi,2,pop.diagnostics['I_N'])
        phi,psi=mc._grouped_scalar(pop,grouped,pop.grouped_oracle,price,aggregate)
        record('group_oracle',phi,psi,2,pop.diagnostics['I_G'])
        feasible=estimated_instrument_space(grouped,b,d)
        result=estimate_joint_giv(grouped,price,aggregate,b,d,feasible,diagnostics=False)
        record('group_feasible',result.phi,result.psi,feasible.combined_rank,feasible.information,
               eigengap=feasible.eigengap,bic_margin=mc._bic_margin(feasible))
        known=estimated_instrument_space(grouped,b,d,combined_rank=2)
        group_weight=projected_weight(known,d)
        reference=grouped@b/(b@b)
        phi=scalar_iv(grouped,group_weight,reference,price)
        psi=scalar_iv(grouped,group_weight,price,aggregate)
        record('group_pca',phi,psi,2,known.information,eigengap=known.eigengap)
        result=estimate_joint_giv(grouped,price,aggregate,b,d,known,diagnostics=False)
        record('group_rank_known',result.phi,result.psi,2,known.information,
               eigengap=known.eigengap,bic_margin=mc._bic_margin(known))
    return rows


def prepare_rng(seed=0):
    """Continue the original seed-0 stream after its two completed cells."""
    rng=np.random.default_rng(seed)
    for n,t in CELLS[:2]:
        for _ in range(mc.REPS):
            rng.standard_normal(t);rng.standard_normal(t);rng.standard_normal((t,n))
    return rng


def complete_draws(draws, sample_sizes, methods, reps):
    expected = pd.MultiIndex.from_product((sample_sizes, methods, range(reps)))
    actual = pd.MultiIndex.from_frame(draws[["T", "method", "replication"]])
    return len(actual) == len(expected) and actual.is_unique and expected.isin(actual).all()


def short_cell_draws(path):
    if path.exists():
        draws = pd.read_pickle(path)
        draws = draws[
            draws.N.eq(mc.N) & draws["T"].isin([100, 1000]) & draws.design.eq("core_weak")
        ].copy()
        if complete_draws(draws, (100, 1000), mc.METHODS, mc.REPS):
            return draws, None
        raise ValueError("incomplete grouped-GIV short-cell checkpoint")
    rng = np.random.default_rng(mc.MC_SEED)
    rows = []
    for n, t in CELLS[:2]:
        population = mc.build_population(n, t, "core_weak")
        rows.extend(mc.simulate_cell(population, mc.REPS, rng))
    return pd.DataFrame(rows), rng


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    pop=mc.build_population(1000,10000,'core_weak')
    if args.check_only:
        started=time.time();result=pd.DataFrame(simulate_cell(pop,1,np.random.default_rng(19))).set_index('method')
        elapsed=time.time()-started
        direct=pd.DataFrame(mc.simulate_cell(pop,1,np.random.default_rng(19))).set_index('method')
        errors={}
        for key in ['phi_hat','psi_hat','phi_se','psi_se','phi_naive_se','psi_naive_se']:
            av=direct[key];bv=result.loc[av.index,key];valid=np.isfinite(av)&np.isfinite(bv)
            errors[key]=float(np.max(np.abs(av[valid]-bv[valid])))
        assert max(errors.values())<1e-8,errors
        audit={'seconds_per_replication':elapsed,'max_differences':errors}
        (RUN_OUTPUT_CACHE/f'{PREFIX}_audit.json').write_text(json.dumps(audit,indent=2))
        print(json.dumps(audit,indent=2),flush=True);return
    short_path = RUN_OUTPUT_CACHE / f"{PREFIX}_short_draws.pkl"
    canonical_path = RUN_OUTPUT_CACHE / "grouped_giv_mc_draws.pkl"
    old, short_rng = short_cell_draws(short_path if short_path.exists() else canonical_path)
    if not short_path.exists() or short_rng is not None:
        atomic_write_pickle(old, short_path)
    checkpoint_path = RUN_OUTPUT_CACHE / f"{PREFIX}_checkpoint.pkl"
    if checkpoint_path.exists():
        status, saved = pd.read_pickle(checkpoint_path)
        completed=int(status['completed'])
        assert status['N']==1000 and status['T']==10000
        assert completed<=mc.REPS
        if not complete_draws(saved, (pop.t,), mc.METHODS, completed):
            raise ValueError("invalid grouped-GIV checkpoint")
        rng=np.random.default_rng()
        rng.bit_generator.state=status['rng_state']
        rows=saved.to_dict('records')
        previous_seconds=float(status['elapsed_seconds'])
    else:
        rng=short_rng if short_rng is not None else prepare_rng()
        rows=[];completed=0;previous_seconds=0.
    started=time.time()
    print(f'Resuming {completed}/{mc.REPS} paired datasets',flush=True)
    for offset in range(completed,mc.REPS,10):
        rows.extend(simulate_cell(pop,min(10,mc.REPS-offset),rng,offset=offset))
        draw=pd.DataFrame(rows)
        status={'N':1000,'T':10000,'replications':mc.REPS,'completed':min(offset+10,mc.REPS),
                'elapsed_seconds':previous_seconds+time.time()-started,'rng_state':rng.bit_generator.state}
        atomic_write_pickle((status, draw), checkpoint_path)
        print(f"FIXED_N_PROGRESS {status['completed']}/{mc.REPS}: {status['elapsed_seconds']:.1f}s",flush=True)
    all_draws=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    summary=pd.concat([mc.summarize(old,1000),mc.summarize(pd.DataFrame(rows),mc.REPS)],ignore_index=True)
    populations=[mc.build_population(n,t,d) for d in mc.DESIGNS for n,t in CELLS]
    designs=mc.design_summary(populations,all_draws)
    atomic_write_pickle(all_draws,RUN_OUTPUT_CACHE/'grouped_giv_mc_draws.pkl')
    atomic_write_pickle(summary,RUN_OUTPUT_CACHE/'grouped_giv_mc_summary.pkl')
    atomic_write_pickle(designs,RUN_OUTPUT_CACHE/'grouped_giv_mc_designs.pkl')
    mc.write_combined_table(summary,OUTPUT_DIR/'table_grouped_giv_mc_combined_tabular.tex')
    print(summary[['N','T','method','psi_bias','psi_sd','psi_coverage','phi_bias','phi_sd','phi_coverage']].to_string(index=False),flush=True)

if __name__=='__main__':main()
