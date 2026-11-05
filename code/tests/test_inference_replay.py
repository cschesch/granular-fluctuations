"""Synthetic checks for paired inference and paper simulation settings."""
from dataclasses import replace
import numpy as np
import inference_mc as mc

TEST_GROUP = "numerical"


def test_paper_simulation_settings():
    from run_inference_mc import simulation_config
    production = simulation_config()
    assert production.reps == 1000
    assert production.n_boot == 199


def test_known_population_certificate():
    # Balanced repeated loadings give an explicit full-panel certificate:
    # the projector onto their orthogonal complement has constant diagonal.
    n=40
    loading=np.tile(np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]),(n//4,1))
    covariance=loading@loading.T+np.eye(n)
    sizes=np.full(n,1/n)
    known=1/n
    value=mc._firm_level_truth(mc.Design('reference_check','reference',n=n,r=2),
                              covariance,sizes,False,known)
    assert abs(value-known)/known < mc.FIRM_GAP_TOL


def test_paired_tolerance_draws():
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
