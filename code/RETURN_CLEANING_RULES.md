# Return cleaning consistency check

Checked 3 November 2026. Scope: the non-network applications. This records the
requested common rule: pooled 1st/99th percentile winsorization for growth and
return outcomes, no clipping of additive flow outcomes, zero-fill internal gaps
of at most three observations in daily series, and no temporal gap filling at
lower frequencies. Endpoints and longer gaps remain missing.

## Current agreement

CPI now uses BLS seasonally adjusted CUSR0000 indexes (32 categories in the
1967-weight panel), with no further seasonal adjustment. The secondary CPI
panel was removed; both employment applications remain.

No significant non-network deviations remain under the agreed observation
calendars. Forbes uses valid observed archive dates as its day calendar: common
missing or corrupted captures are skipped. Personal internal gaps of at most
three observed dates are filled by carrying wealth forward, yielding zero growth
during the gap and the full catch-up change on reappearance. Longer or endpoint
gaps exclude the person. Changes are between successive retained captures,
without normalization by elapsed calendar days.

The BEA–BLS sector TFP comparison without network propagation is treated as a
network application for these cleaning rules. It remains unwinsorized and is
outside this audit's non-network scope.

All active non-network growth/return panels use pooled 1/99 winsorization.
The CPI panel follows that rule; this still means raw CPI additivity does
not survive clipping. Annual, quarterly, monthly, and election-growth outcomes
have no local temporal gap bridging.

## Rules that agree, and distinctions that matter

- Daily mutual-fund aggregation uses December 31, 2010 AUM shares. Monthly
  fund flows enter in levels without detrending, retaining mean-AUM size
  normalization and no magnitude screen. Mean AUM uses the January 2011–December
  2025 flow-observation months. Portfolios with zero AUM throughout that window
  have no positive normalization and are excluded. Neither fund application fits a trend.
- Daily common-stock returns and daily mutual-fund returns bridge internal runs
  of up to three **trading dates** with zero returns. They do not bridge
  weekends, exchange holidays, endpoint gaps, or part of a longer missing run.
  Common-stock returns now receive pooled 1/99 winsorization and grand-mean
  centering after gap filling; the earlier audit's no-winsorization exception
  is resolved.
- The mutual-fund rule applies after consolidating observed classes into each
  portfolio's daily return. Missing class returns can be bypassed using
  observed siblings on the **same day**, with weights renormalized.
- Monthly class returns used to calculate fund flows can use observed siblings
  in the **same month**. This is cross-sectional substitution, not bridging
  a missing month from other months. A positive-AUM month with no observed
  class return excludes the fund from the balanced flow panel.
- Documented inactive AUM, no-issuance quarters, and zero patent/drug counts are
  structural zeros in flow construction, not temporal interpolation. Flows
  are not detrended or winsorized. Bonds remove issuer-specific quarter-of-year
  means, using mean raw issuance as fixed size. Other flow panels retain raw
  levels and zeros without seasonal adjustment.
- Annual employment held fixed within a year in quarterly productivity is a
  mixed-frequency input convention, not filling missing quarterly growth.

## Country-equity timing correction

The preserved World Bank extract has no 1974 capitalization observations; the
WDI indicator's published coverage begins in 1975. The estimation loader now
starts returns in **January 1976**, keeping fixed end-1975 capitalization and
12 markets. The prepared raw French return file still starts in 1975 so no
source history is discarded. The estimation window is 600 months through
December 2025.

Source: [World Bank capitalization series](https://data.worldbank.org/indicator/CM.MKT.LCAP.CD).

## Flow weights

For flows f_it and a positive fixed size s_i, define y_it=f_it/s_i
and S_i=s_i/sum_j(s_j). The grouped weighted outcome is

    sum_{i in B} S_i y_it = sum_{i in B} f_it / sum_j(s_j).

The implementation keeps flow levels (seasonally adjusted for bonds) in the
source panels, divides
them by s_i at estimation, and uses S_i for both LPT grouping and aggregation.
The shared grouping routine now has no separate raw-sum mode. This is
equivalent to the previous raw sum up to one common constant, which cancels
from the variance share. Multiplying raw flows by s_i again would double-weight
size and change the aggregate.

The current positive sizes are sample-mean issuance/count/value for nonnegative
flow panels and mean portfolio AUM for signed mutual-fund flows. The latter
avoids negative or zero average net flows as sizes. Both the LPT grouping and
the equivalent size-weighted relative-flow representation use those same
sizes. The implementation now makes that size normalization explicit; the economic
aggregate and grouping rule are preserved.
