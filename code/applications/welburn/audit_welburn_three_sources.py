"""Inspect three WRDS linkage sources without changing the Welburn application.

Licensed identities and edge overlays stay in .cache/firm_networks. Reports in
output contain only aggregate counts. The acquisition helper accepts an open
connection and never logs in. This script is entirely offline.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from applications.welburn.prepare_welburn_links import normalize_name
from applications.welburn.welburn_network import _firm_means
from wrds_inputs import wrds_path

from project_paths import ROOT
RAW = ROOT / 'data/welburn2020/wrds_inputs'
PRIVATE = ROOT / 'code/.cache/firm_networks/welburn_three_sources'
WINDOWS = {'annual': ('2015-01-01', '2019-12-31'),
           'paper': ('2015-01-01', '2019-03-31')}


def latest(stem):
    paths = sorted(RAW.glob(stem + '_*.csv.gz'))
    return paths[-1] if paths else None


def ids(series):
    return series.astype('string').str.replace(r'\.0$', '', regex=True).str.zfill(6)


def edges(frame, universe, left='gvkey', right='customer_gvkey'):
    frame = frame[[left, right]].rename(columns={left: 'gvkey', right: 'customer_gvkey'}).copy()
    for key in frame:
        frame[key] = ids(frame[key])
    return frame.loc[frame.gvkey.isin(universe) & frame.customer_gvkey.isin(universe)
                     & frame.gvkey.ne(frame.customer_gvkey)].drop_duplicates().reset_index(drop=True)


def pairset(frame):
    return set(frame.itertuples(index=False, name=None))


def moments(frame, universe):
    counts = frame.gvkey.value_counts().add(frame.customer_gvkey.value_counts(), fill_value=0)
    degree = counts.reindex(universe, fill_value=0)
    return dict(nodes=len(universe), directed_edges=len(frame),
                mean_total_degree=float(degree.mean()), median_total_degree=float(degree.median()),
                maximum_total_degree=int(degree.max()), isolated_firms=int(degree.eq(0).sum()))


def date_window(frame, label, column='srcdate'):
    lo, hi = WINDOWS[label]
    return frame.loc[frame[column].between(pd.Timestamp(lo), pd.Timestamp(hi))]


def source1(base, source, universe):
    source = source.copy()
    source['srcdate'] = pd.to_datetime(source.srcdate)
    source['customer_gvkey'] = ids(source.cgvkey)
    source['gvkey'] = ids(source.gvkey)
    source['customer_key'] = source.cnms.map(normalize_name)
    keys = ['gvkey', 'srcdate', 'cid', 'sid', 'customer_key']
    for frame in (base, source):
        for key in ['cid', 'sid']:
            frame[key] = pd.to_numeric(frame[key], errors='coerce').astype('Int64')
    assignments = source[keys + ['customer_gvkey']].drop_duplicates()
    ambiguity = assignments.groupby(keys, dropna=False).customer_gvkey.transform('nunique')
    unique = assignments.loc[ambiguity.eq(1)].drop_duplicates(keys)
    matched = base.merge(unique, on=keys, how='left', suffixes=('', '_wrds'), validate='many_to_one')
    conflict = (matched.customer_gvkey.notna() & matched.customer_gvkey_wrds.notna()
                & matched.customer_gvkey.ne(matched.customer_gvkey_wrds)).fillna(False)
    rescued = matched.customer_gvkey.isna() & matched.customer_gvkey_wrds.notna()
    matched['source1_rescued'] = rescued
    matched['source1_conflict'] = conflict
    matched['customer_gvkey_after_ids'] = matched.customer_gvkey.fillna(matched.customer_gvkey_wrds)
    known = matched.customer_gvkey.notna()
    pd.testing.assert_series_equal(matched.loc[known, 'customer_gvkey_after_ids'],
                                   matched.loc[known, 'customer_gvkey'], check_names=False)
    matched.to_parquet(PRIVATE / 'supply_chain_ids_row_comparison.parquet', index=False)
    report = dict(raw_rows=len(source), ambiguous_assignment_rows=int(ambiguity.gt(1).sum()),
                  rule='Fill only previously missing customer IDs on exact raw-record keys; preserve existing assignments. Raw source unions are diagnostics, not the accepted overlay.',
                  windows={})
    for label in WINDOWS:
        b, s, m = date_window(base, label), date_window(source, label), date_window(matched, label)
        old = edges(b, universe)
        direct = edges(s, universe)
        union = pd.concat([old, direct]).drop_duplicates()
        row_overlay = edges(m, universe, right='customer_gvkey_after_ids')
        union.to_parquet(PRIVATE / f'ids_union_edges_{label}.parquet', index=False)
        report['windows'][label] = dict(
            raw_wrds_rows=len(s), baseline=moments(old, universe),
            wrds_direct=moments(direct, universe), union=moments(union, universe),
            matching_record_overlay=moments(row_overlay, universe),
            common_edges=len(pairset(old) & pairset(direct)),
            new_edges_in_union=len(pairset(direct) - pairset(old)),
            rescued_raw_rows=int(m.source1_rescued.sum()),
            conflicting_raw_rows=int(m.source1_conflict.sum()),
            rescued_usable_rows=int((m.source1_rescued & m.usable_name).sum()))
    return report, matched


def main(*, quiet=False):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / 'code/.cache/firm_networks/customer_name_matches.parquet'
    base = pd.read_parquet(base_path)
    base['srcdate'] = pd.to_datetime(base.srcdate)
    base['gvkey'] = ids(base.gvkey)
    base['customer_gvkey'] = ids(base.customer_gvkey)
    universe = _firm_means().index
    report = dict(scope='Three-source matching audit; canonical network and estimator unchanged',
                  windows=WINDOWS, financial_firms=len(universe), inputs={})
    source_path = latest('wrds_supply_chain_ids')
    if source_path:
        report['inputs']['supply_chain_ids'] = str(source_path.relative_to(ROOT))
        report['source1'], matched = source1(base, pd.read_csv(source_path,
            dtype={'gvkey': 'string', 'cgvkey': 'string'}), universe)
    subsidiary_path = latest('wrds_company_subsidiaries')
    collision_path = latest('wrds_subsidiary_global_aliases')
    if subsidiary_path and collision_path:
        from applications.welburn.welburn_subsidiary_matching import analyze_subsidiaries
        report['inputs']['subsidiaries'] = str(subsidiary_path.relative_to(ROOT))
        report['inputs']['subsidiary_global_collision_check'] = str(collision_path.relative_to(ROOT))
        report['source2'], final_matched = analyze_subsidiaries(
            matched, subsidiary_path, universe, PRIVATE, collision_path=collision_path)
    access_path = RAW / 'access_2026-11-03.json'
    if access_path.exists():
        report['access'] = json.loads(access_path.read_text())
        report['source3'] = dict(name='FactSet Revere Supply Chain / Relationships',
            status='permission_denied', rows_downloaded=0,
            reason='Actual SELECT denied at schema level; catalog table SELECT flags alone do not establish access.')
    financial_path = wrds_path('compustat_annual')
    report['inputs']['existing_annual_financials'] = str(financial_path.relative_to(ROOT))
    report['input_sha256'] = {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                              for name, path in report['inputs'].items()}
    report['baseline_cache_sha256'] = hashlib.sha256(base_path.read_bytes()).hexdigest()
    target = ROOT / 'output/welburn_three_source_matching_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    pd.to_pickle(report, ROOT / 'code/.cache/run_outputs/welburn_three_source_matching_audit.pkl')
    lines = ['# Welburn: three-source matching audit', '',
        'All counts use the same 6,296-firm financial universe. Existing customer IDs are preserved; the matching overlay is saved separately from the application.', '',
        '| Cumulative matching stage | 2015–2019 links | 2015Q1–2019Q1 links |',
        '|---|---:|---:|']
    s = report['source1']['windows']
    lines += [f"| Existing reconstruction | {s['annual']['baseline']['directed_edges']:,} | {s['paper']['baseline']['directed_edges']:,} |",
              f"| Add WRDS Supply Chain with IDs | {s['annual']['matching_record_overlay']['directed_edges']:,} | {s['paper']['matching_record_overlay']['directed_edges']:,} |"]
    if 'source2' in report:
        s2 = report['source2']['windows']
        lines += [f"| Add dated Exhibit 21 subsidiary matches | {s2['annual_2015_2019']['after']['directed_edges']:,} | {s2['paper_2015Q1_2019Q1']['after']['directed_edges']:,} |"]
    lines += ['', 'FactSet Revere: account lacks schema access; no relationship rows were downloaded.', '',
        'The original paper reports 20,477 observed links across a different 5,821-firm universe. These counts do not establish recovery of the same links.', '',
        'Existing conflicting assignments are retained and separately flagged. Subsidiary names require exact normalized matches, a unique contemporaneous parent across the global Exhibit 21 extract, and a filing no older than 550 days. Filing dates establish information availability rather than exact acquisition/disposal dates.', '',
        'No network completion, flow allocation, shock recovery, or MGP re-estimation is performed in this audit.']
    (ROOT / 'output/welburn_three_source_matching_report.md').write_text('\n'.join(lines) + '\n')
    if not quiet:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
