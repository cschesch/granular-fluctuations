#!/usr/bin/env python3
"""Checks for redraw selection and reuse of tolerance-bootstrap draws."""
import os
os.environ['NUMBA_NUM_THREADS']='2'
import numpy as np
from dataclasses import replace
import inference_mc as mc
from mgp_core import RUN_OUTPUT_CACHE


def main():
    from run_mc_new import simulation_config
    production=simulation_config()
    assert production.reps==1000
    assert production.n_boot==199
    # Replaying the selection preserves the original idiosyncratic draw and
    # reproduces the explicitly verified second loading draw exactly.
    saved=np.load(RUN_OUTPUT_CACHE/'selected_nonregular_zipf_r3.npz')
    rng=np.random.default_rng(20260901)
    variance=np.exp(.5*rng.standard_normal(1000));variance/=variance.mean()
    rng.standard_normal((1000,3))
    raw=rng.standard_normal((1000,3))
    assert np.array_equal(variance,saved['sigma2_e'])
    assert np.array_equal(raw,saved['raw_loadings'])
    # Balanced repeated loadings give an explicit full-panel certificate:
    # the projector onto their orthogonal complement has constant diagonal.
    n=200
    loading=np.tile(np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]),(n//4,1))
    covariance=loading@loading.T+np.eye(n)
    sizes=np.full(n,1/n)
    known=1/n
    value=mc._firm_level_truth(mc.Design('reference_check','reference',n=n,r=2),
                              covariance,sizes,False,known)
    assert abs(value-known)/known < mc.FIRM_GAP_TOL
    original=mc.build_population(mc.Design('baseline_check','baseline',seed=20260901))
    assert original.regular is True
    assert mc._population_regularity(saved['sigma_grouped_normalized']) is False
    d=mc.Design('test_mc_new','test',n=40,m=5,r=2,t=40,seed=123)
    cfg=mc.McConfig(reps=12,n_boot=19,n_subsample=8,seed=0)
    common=mc.run_design(d,cfg,resampling_families=('bootstrap',),extra_delta_scales=(.5,2.))
    assert len(common['share'])==len(common['se'])==cfg.reps
    assert common['row']['reps']==common['row']['reps_resampling']==cfg.reps
    for item in common['coverage']:
        if np.isfinite(item['cover_upper_estimand']):
            assert item['reps']==cfg.reps
    normal=next(x for x in common['coverage'] if x['method']=='normal')
    upper=common['share']+mc.Z95*common['se']
    assert normal['cover_upper_estimand']==np.mean(common['row']['gamma_bar0']<=upper)
    for scale in (.5,2.):
        separate=mc.run_design(replace(d,delta_scale=scale),cfg,resampling_families=('bootstrap',))
        wanted=next(x for x in separate['coverage'] if x['method']=='gradient_bootstrap')
        got=common['tolerance_coverage'][scale][0]
        for field in ('cover_upper_estimand','cover_upper_truth','median_upper','median_length'):
            assert abs(got[field]-wanted[field])<1e-10,(scale,field,got[field],wanted[field])
        assert np.array_equal(common['share'],separate['share'])
    print('PASS: all methods use the same replication count; selected draw replay and paired tolerance runs match exactly.')

if __name__=='__main__':main()
