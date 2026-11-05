#!/usr/bin/env python3
"""Figures and merged diagnostics table for the revised Monte Carlo appendix.

Consumes run_design results, never runs estimators or edits manuscript files.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from mgp_core import OUTPUT_DIR as OUT, RUN_OUTPUT_CACHE as CACHE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm

METHODS=('normal','bootstrap','gradient_bootstrap','subsampling')
LABELS={'normal':'Normal','bootstrap':'Naive bootstrap',
        'gradient_bootstrap':'Directional bootstrap','subsampling':'Subsampling'}
COLORS={'normal':'tab:red','bootstrap':'tab:orange',
        'gradient_bootstrap':'tab:blue','subsampling':'tab:green'}
MARKERS={'normal':'^','bootstrap':'s','gradient_bootstrap':'o','subsampling':'D'}
TGRID=(50,100,250,1000,2500,10000)


def setup():
    plt.style.use('default')
    plt.rcParams.update({'font.size':12,'axes.titlesize':13,'legend.fontsize':10,
                         'pdf.fonttype':42,'savefig.facecolor':'white'})


def save(fig,stem):
    fig.tight_layout()
    fig.savefig(OUT/f'{stem}.pdf',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(OUT/f'{stem}.pdf',flush=True)


def frames(results):
    rows=pd.DataFrame([r['row'] for r in results])
    rows['label']=rows['label'].str.replace('Non-regular redraw','Non-regular',regex=False)
    cov=pd.DataFrame([c for r in results for c in r.get('coverage',[])])
    if not cov.empty:
        extra=[c for c in ('r','T','rho','regular','delta_scale','gamma_bar0') if c not in cov]
        cov=cov.merge(rows[['design',*extra]],on='design',how='left',validate='many_to_one')
    draws={r['row']['design']:np.asarray(r['standardized']) for r in results if 'standardized' in r}
    return rows,cov,draws


def regimes(rows,meta):
    pairs=[]
    for regular,key in ((True,'regular_r'),(False,'nonregular_r')):
        r=meta.get(key)
        if r is None:
            selected=rows[rows.regular.astype(bool).eq(regular)]
            if selected.empty:
                raise ValueError(f'No {key} in sweep; provide meta explicitly')
            r=int(selected.r.iloc[0])
        pairs.append((int(r),regular))
    return pairs


def panel_label(index,regular):
    return f"({'a' if index==0 else 'b'}) {'Regular case' if regular else 'Non-regular case'}"


def t_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(TGRID,[f'{t:,}' for t in TGRID],fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel('T')


def coverage(results,meta):
    rows,cov,_=frames(results)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.1),sharey=True)
    for j,(ax,(r,regular)) in enumerate(zip(axes,regimes(rows,meta))):
        for method in METHODS:
            d=cov[cov.r.eq(r)&cov.regular.eq(regular)&cov.method.eq(method)].sort_values('T')
            ax.plot(d['T'],d.cover_upper_estimand,
                    color=COLORS[method],marker=MARKERS[method],ms=4.4,lw=1.6,label=LABELS[method])
        ax.axhline(.95,color='black',lw=1,ls='--')
        t_axis(ax); ax.set_ylim(0,1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_title(panel_label(j,regular))
    axes[0].set_ylabel(r'Coverage of $\tilde\gamma$ by the 95% UCL')
    axes[0].legend(loc='lower right',ncol=1)
    save(fig,'mc_new_coverage')


def preferred(rows,cov):
    """Use the block directional procedure for serially dependent designs."""
    selected=[]
    for row in rows.itertuples():
        method='block_gradient_bootstrap' if row.rho else 'gradient_bootstrap'
        one=cov[cov.design.eq(row.design)&cov.method.eq(method)]
        if len(one)!=1:
            raise ValueError(f'{row.design}: missing {method}')
        selected.append(one.iloc[0][['design','cover_upper_estimand','cover_upper_truth','median_upper']])
    return rows.merge(pd.DataFrame(selected),on='design',validate='one_to_one')


def designs(results):
    rows,cov,_=frames(results)
    d=preferred(rows,cov)
    pos=np.arange(len(d))
    fig,axes=plt.subplots(1,3,figsize=(12.7,max(5.6,.30*len(d)+1.2)),sharey=True,
                          gridspec_kw={'width_ratios':[1,1,1]})
    for ax,col,title in zip(axes[:2],('cover_upper_estimand','cover_upper_truth'),
                            (r'(a) Coverage of $\tilde\gamma$',r'(b) Coverage of $\bar\gamma$')):
        ax.scatter(d[col],pos,color='tab:blue',s=27,zorder=3,clip_on=False)
        ax.axvline(.95,color='black',ls='--',lw=1)
        ax.set_xlim(.70,1.)
        ax.set_xticks(np.arange(.70,1.001,.05))
        ax.set_xlabel('Coverage of the 95% UCL',fontsize=13)
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_title(title,fontsize=14)
    ax=axes[2]
    for i,row in enumerate(d.itertuples()):
        ax.plot([row.gamma0,row.median_upper],[i,i],color='.7',lw=1.1,zorder=1)
    # Larger diamond underneath the smaller circle: coincident values remain visible.
    ax.scatter(d.gamma0,pos,color='tab:green',marker='D',s=53,zorder=3,
               label=r'Full-panel ceiling $\bar\gamma$')
    ax.scatter(d.median_upper,pos,color='lightskyblue',marker='o',s=35,zorder=4,
               label='Median 95% UCL')
    ax.scatter(d.gamma_bar0,pos,color='tab:blue',marker='o',s=28,zorder=5,
               label=r'Grouped ceiling $\tilde\gamma$')
    ax.set_xlim(0,.15)
    ax.set_xticks(np.arange(0,.151,.03))
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_xlabel('Share of aggregate variance',fontsize=13)
    ax.set_title('(c) Ceilings and median UCL',fontsize=14)
    handles,labels=ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in (0,2,1)],[labels[i] for i in (0,2,1)],
              loc='lower right',fontsize=9)
    axes[0].set_yticks(pos,d.label,fontsize=13)
    axes[0].set_ylim(len(d)-.4,-.6)
    for ax in axes:
        ax.grid(axis='x',color='.9',lw=.6)
        ax.set_axisbelow(True)
    save(fig,'mc_new_designs')
    return d


def qq(results,meta):
    rows,_,draws=frames(results)
    # Normalize by the realized MC standard deviation, retaining population centering.
    # This avoids selecting an arbitrary dual-branch variance at a nonregular point.
    for result in results:
        if 'share' in result:
            values=np.asarray(result['share'])
            draws[result['row']['design']]=(values-result['row']['gamma_bar0'])/values.std(ddof=1)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.2),sharey=True)
    for j,(ax,(r,regular)) in enumerate(zip(axes,regimes(rows,meta))):
        for t,color in ((100,'tab:orange'),(1000,'tab:blue'),(10000,'tab:green')):
            name=rows[rows.r.eq(r)&rows.regular.eq(regular)&rows['T'].eq(t)].design.iloc[0]
            z=np.sort(draws[name]); quantiles=norm.ppf((np.arange(len(z))+.5)/len(z))
            ax.plot(quantiles,z,'o',ms=2.5,alpha=.8,color=color,label=f'$T={t:,}$')
        lim=4
        ax.plot([-lim,lim],[-lim,lim],color='black',ls='--',lw=1)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_xlabel('Standard normal quantile')
        ax.set_title(panel_label(j,regular))
        ax.legend(loc='upper left')
    axes[0].set_ylabel(r'$(\hat{\tilde\gamma}-\tilde\gamma)/s_{\mathrm{MC}}$')
    save(fig,'mc_new_limit_law')


def bias(results,meta):
    rows,_,_=frames(results)
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    for r,regular in regimes(rows,meta):
        d=rows[rows.r.eq(r)&rows.regular.eq(regular)].sort_values('T')
        color='tab:green' if regular else 'tab:red'
        label='Regular case' if regular else 'Non-regular case'
        ax.plot(d['T'],100*d.bias.abs(),color=color,marker='o' if regular else 's',lw=1.7,label=label)
        power=1 if regular else .5
        anchor=float(d.bias.abs().iloc[0])
        t0=float(d['T'].iloc[0])
        grid=d['T'].to_numpy(float)
        ax.plot(grid,100*anchor*(t0/grid)**power,color=color,ls='--',lw=1,
                label=r'$T^{-1}$' if regular else r'$T^{-1/2}$')
    t_axis(ax); ax.set_yscale('log')
    ax.set_ylabel('Absolute bias (percentage points)')
    ax.legend(ncol=2)
    save(fig,'mc_new_bias')


def tolerance(sweep_results,delta_results,meta):
    rows,cov,_=frames([*sweep_results,*delta_results])
    grad=cov[cov.method.eq('gradient_bootstrap')]
    d=rows.merge(grad[['design','cover_upper_estimand','median_upper']],on='design')
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.2))
    styles={.5:(':','v'),1.:('-','o'),2.:('--','^')}
    for r,regular in regimes(rows,meta):
        color='tab:green' if regular else 'tab:red'
        for c,(style,marker) in styles.items():
            selected=d[d.r.eq(r)&d.regular.eq(regular)&np.isclose(d.delta_scale,c)].sort_values('T')
            label=rf'{"Regular" if regular else "Non-regular"}, $c={c:g}$'
            for ax,col in zip(axes,('cover_upper_estimand','median_upper')):
                ax.plot(selected['T'],selected[col],color=color,ls=style,marker=marker,ms=3.5,lw=1.4,label=label)
    axes[0].axhline(.95,color='black',ls='--',lw=1)
    axes[0].set_ylabel(r'Coverage of $\tilde\gamma$')
    axes[0].set_title('(a) Coverage of the 95% UCL')
    axes[0].margins(y=.08)
    ceilings=rows[['regular','gamma_bar0']].drop_duplicates()
    for row in ceilings.itertuples():
        label=('Regular' if row.regular else 'Non-regular')+' ceiling'
        axes[1].axhline(row.gamma_bar0,color='black',ls='--',lw=1.2,label=label)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('Median 95% UCL')
    axes[1].set_title('(b) Reported ceiling')
    for ax in axes:
        t_axis(ax)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[0].legend(ncol=2,loc='lower right',fontsize=9)
    handles,labels=axes[1].get_legend_handles_labels()
    order=(0,1,2,6,3,4,5,7)
    axes[1].legend([handles[i] for i in order],[labels[i] for i in order],
                   loc='lower left',ncol=2,fontsize=9)
    save(fig,'mc_new_delta')


def write_table(d):
    assert np.array_equal(d.reps.to_numpy(),d.reps_resampling.to_numpy())
    lines=['% Generated by code/plot_mc_new.py; do not edit.',
           f'% Paired replications per method and design: {int(d.reps.iloc[0])}.',
           r'{\footnotesize\setlength{\tabcolsep}{2pt}',
           r'\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}rrcrrrrrr@{}}',
           r'\toprule',
           r'Design & $\bar\gamma$ & $\tilde\gamma$ & Regular & Bias (pp) & SD (pp) & Skew & Ex. kurt. & Coverage of $\tilde\gamma$ & Coverage of $\bar\gamma$\\',
           r'\midrule']
    for row in d.itertuples():
        regular=r'$\checkmark$' if row.regular else ''
        lines.append(f'{row.label} & {100*row.gamma0:.2f} & {100*row.gamma_bar0:.2f} & '
                     f'{regular} & {100*row.bias:+.2f} & {100*row.sd_share:.2f} & '
                     f'{row.skew:+.2f} & {row.excess_kurtosis:+.2f} & '
                     f'{100*row.cover_upper_estimand:.1f} & {100*row.cover_upper_truth:.1f}'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular*}}']
    (OUT/'table_mc_new_tabular.tex').write_text('\n'.join(lines)+'\n')


def compustat():
    from mgp_core import subsampling_summary
    row=pd.read_pickle(CACHE/'compustat_qtr_summary.pkl').iloc[0]
    estimate=float(row.mgp_share)
    sub=row
    if pd.isna(row.get('subsample_p05',np.nan)):
        sub=subsampling_summary(pd.read_pickle(CACHE/'compustat_qtr_draws_subsampling.pkl'),estimate,int(row['T']))
    specs=[('Normal approximation',estimate-1.6448536269514722*row.se_share,row.normal_ucl95),
           ('Naive bootstrap',row.bootstrap_p05,row.bootstrap_p95),
           ('Block naive bootstrap',row.moving_block_bootstrap_p05,row.moving_block_bootstrap_p95),
           ('Directional bootstrap',row.gradient_bootstrap_p05,row.gradient_bootstrap_p95),
           ('Block directional bootstrap',row.moving_block_gradient_bootstrap_p05,row.moving_block_gradient_bootstrap_p95),
           ('Subsampling',sub['subsample_p05'],sub['subsample_p95'])]
    fig,ax=plt.subplots(figsize=(7.8,3.6))
    for i,(label,low,high) in enumerate(specs):
        ax.hlines(i,100*low,100*high,color='tab:blue',lw=2)
        ax.vlines([100*low,100*high],i-.12,i+.12,color='tab:blue',lw=1.2)
        ax.scatter(100*estimate,i,color='tab:blue',s=30,zorder=4)
    ax.axvline(100*estimate,color='.5',lw=.8,ls=':',zorder=1)
    ax.set_yticks(np.arange(len(specs)),[s[0] for s in specs])
    ax.set_ylim(len(specs)-.45,-.55)
    ax.set_xlim(0,max(s[2] for s in specs)*100*1.06)
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.set_xlabel('Granularity ceiling')
    save(fig,'mc_new_compustat_intervals')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CACHE/'mc_new_results.pkl')
    section = parser.add_mutually_exclusive_group()
    section.add_argument('--compustat-only', action='store_true')
    section.add_argument('--simulation-only', action='store_true')
    args=parser.parse_args(); setup()
    if args.compustat_only:
        compustat(); return
    with args.input.open('rb') as handle:
        data=pickle.load(handle)
    meta=data.get('meta',{})
    coverage(data['sweep'],meta)
    merged=designs(data['main'])
    qq(data['sweep'],meta)
    bias(data['sweep'],meta)
    tolerance(data['sweep'],data['delta'],meta)
    write_table(merged)
    if not args.simulation_only:
        compustat()

if __name__=='__main__':
    main()
