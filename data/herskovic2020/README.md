# Herskovic et al. (2020)

This folder contains code and public inputs from the replication package for
*Firm Volatility in Granular Networks*. Source information is in `source.json`.
Licensed Compustat/CRSP observations and restricted linkage inputs must be
obtained separately.

Our analysis code is in `code/applications/herskovic/`. Its preparation step
requires MATLAB with Statistics and Machine Learning Toolbox and reproduces
the authors' 5,000-firm model-size draw with seed 100. Set `MATLAB` to the
executable path when it is not on PATH.

From the repository root, with the Python environment activated:

```sh
cd code
python -m applications.herskovic.prepare
```

The MATLAB exporter is alongside that preparation script. Generated sizes,
networks, and recovered shocks are stored under `code/.cache/`.

## Disclosed sales shares in the completed network

The completion uses Appendix A's structural normalization. The normalized
network has row sums of one, and a supplier's raw sales share to customer
`j` is `gamma * W[i,j]`. A disclosed share `q[i,j]` therefore fixes
`W[i,j] = q[i,j] / gamma`. The remaining row weight is
`1 - sum(q[i,j]) / gamma`, allocated across simulated customers using the
calibrated size rule. Rows with disclosed totals above `gamma` raise an error.
The disclosure screen on additional public customers applies to raw sales
shares, requiring `gamma * W[i,j] < 0.10`.

This is an explicit structural interpretation for our empirical completion.
The paper's Section 4.6 discusses qualifications to interpreting its fitted
reduced-form propagation coefficient as the intermediate-sales fraction.
Network checks verify disclosed raw shares in `gamma * W` and normalization
in `W`. The full propagation operator is `I - gamma * W`; Schur reduction
sets primitive shocks outside the balanced panel to zero and retains their
transmission effects.
