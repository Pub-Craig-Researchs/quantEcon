# +quantecon Package

High-performance MATLAB econometrics toolbox combining best-in-class methods from:

- `MFE Toolbox` (Financial Econometrics)
- `FSDA` (Robust Statistics)
- `BEAR Toolbox` (Bayesian VARs)
- [`DiDToolbox`](https://github.com/relsas/DiDToolbox) (Difference-in-Differences)
- `panelPlus` (Causal Inference & ML)
- `PSY-IVX` (Bubble Detection & Predictive Regression)
- `Abadie & Imbens` (Matching Estimators & Bias Correction)
- `Quilis` (Temporal Disaggregation)
- `Montiel Olea, Stock & Watson` (Local Projection & SVAR-IV Inference)
- `Miranda-Agrippino & Banbura-Modugno` (Dynamic Factor Models)
- `Dube, Girardi, Jorda & Taylor` (LP-DiD)
- `Ballestra & Tezza` (Commodity Futures Pricing)

> **20 modules** · **180+ functions** · Cross-validated against Stata, Python & R

## Installation

```matlab
addpath('path/to/containing/folder');
savepath;
```

## Module Overview

| #  | Module            | Description                                                                                               | Functions |
| -- | ----------------- | --------------------------------------------------------------------------------------------------------- | --------- |
| 1  | `+base`         | OLS, GMM, HAC, KDE, Bootstrap (Wild/Block/Stationary), Distributions, HP Filter, VARHAC, Matrix Ops       | 17        |
| 2  | `+panel`        | Fixed Effects, Clustering, DiD, Panel VAR                                                                 | 12        |
| 3  | `+timeseries`   | GARCH, SVAR, SVAR-IV, LP (LagAug/GLS/Smooth/Bayesian), SimBands, Markov, ARFIMA, fOU, DEKF, Hurst, Disagg | 43        |
| 4  | `+bayes`        | BVAR, TVP-VAR, SSVS, FAVAR, MF-BVAR, Sign-ID, MH, HPD, Gauss-Hermite                                      | 16        |
| 5  | `+tests`        | ADF, DF-GLS, PSY, IVX, MCS, BSDS/SPA, TV-Granger, Homosked, Verification Suites                           | 25        |
| 6  | `+multivariate` | DCC, BEKK, Copula, DCC-MIDAS, Reduced Rank                                                                | 5         |
| 7  | `+finance`      | Connectedness, Networks, Heston, HistVol, Indicators, CommodityFutures                                    | 7         |
| 8  | `+risk`         | CoVaR, MES, SRISK                                                                                         | 3         |
| 9  | `+ml`           | Causal Forest (Honest/DR/QTE), Double ML                                                                  | 2         |
| 10 | `+causal`       | RDD, PSM, SEM, LP-DiD                                                                                     | 4         |
| 11 | `+factor`       | IPCA, Scaled PCA, DFM, Block-DFM                                                                          | 4         |
| 12 | `+robust`       | LTS, MCD, Robust PCA, Trimmed Clustering                                                                  | 5         |
| 13 | `+midas`        | MIDAS Regression (Beta/ExpAlmon)                                                                          | 3         |
| 14 | `+forecasting`  | BMA, Murphy Diagram                                                                                       | 2         |
| 15 | `+models`       | QBLL-VAR                                                                                                  | 1         |
| 16 | `+sem`          | PLS-SEM                                                                                                   | 1         |
| 17 | `+optimization` | Simplex, CsminWel, NumGrad                                                                                | 3         |
| 18 | `+utils`        | VAR Utilities                                                                                             | 1         |
| 19 | `+vis`          | Gramm, exportFig, ConnectednessVis                                                                        | 17        |
| 20 | `+dsge`         | Gensys, Perturbation, Projection, Kalman, Tauchen, Rouwenhorst, Lyapunov                                  | 9         |

---

## API Reference

### 1. `+base` — Core Utilities

#### Ols

OLS with optional robust (HC1) standard errors.

```matlab
results = quantecon.base.Ols(y, X);
results = quantecon.base.Ols(y, X, 'Robust', true);
```

#### GmmFit

Generalized Method of Moments with Newey-West HAC.

```matlab
results = quantecon.base.GmmFit(moment_func, theta0, Y, X, Z);
```

#### DieboldMariano

Test for equal predictive accuracy (Diebold & Mariano 1995).

```matlab
res = quantecon.base.DieboldMariano(e1, e2);
res = quantecon.base.DieboldMariano(e1, e2, 'LossType', 'MAE', 'Lag', 4);
```

#### ClarkWest

Out-of-sample predictive accuracy for nested models (Clark & West 2007).

```matlab
res = quantecon.base.ClarkWest(actual, forecast_restricted, forecast_unrestricted);
```

#### NeweyWest

Newey-West (1987) HAC standard errors.

```matlab
[V, se] = quantecon.base.NeweyWest(X, residuals, 5);
```

#### HacCov

HAC covariance with optional prewhitening.

```matlab
[hac, lags] = quantecon.base.HacCov(u, 'Lags', 4, 'Prewhite', true);
```

#### EmpiricalCdf

Transform data to uniform margins [0, 1] for copula modeling.

```matlab
u = quantecon.base.EmpiricalCdf(DataMatrix);
```

#### WildBootstrap

Wild Bootstrap resampling (Rademacher weights).

```matlab
boot_samples = quantecon.base.WildBootstrap(residuals, 1000);
```

#### BlockBootstrap

Circular block bootstrap for stationary, dependent time series (Kunsch 1989; Politis & Romano 1992).

```matlab
[bsdata, idx] = quantecon.base.BlockBootstrap(y, 1000, 10);
% For multivariate: pass indices, then index your (T x K) matrix
[~, idx] = quantecon.base.BlockBootstrap((1:T)', B, w);
```

#### StationaryBootstrap

Politis & Romano (1994) stationary bootstrap with random (geometric) block lengths.

```matlab
[bsdata, idx] = quantecon.base.StationaryBootstrap(y, 1000, 10);
```

#### Distributions

Wishart and Inverse-Wishart draws for Bayesian inference.

```matlab
A = quantecon.base.Distributions.wish(H, nu);
A = quantecon.base.Distributions.iwish(H, nu);
```

#### StatTools

Descriptive statistics (mean, variance, skewness, kurtosis).

```matlab
stats = quantecon.base.StatTools.moments(x);
```

#### Utils

Vector/matrix helpers: `colvec`, `rowvec`, `add_const`, `lag`.

```matlab
x = quantecon.base.Utils.colvec(x);
X = quantecon.base.Utils.add_const(X);
```

#### Kde

Kernel Density Estimation with automatic bandwidth selection.

```matlab
mdl = quantecon.base.Kde(x, 'Kernel', 'gaussian', 'Bandwidth', 'silverman');
[f, xi] = mdl.evaluate();       % evaluate on auto grid
f_new = mdl.evaluate(xi_new);   % evaluate at specific points
```

#### HpFilter

Hodrick-Prescott (1997) filter for trend-cycle decomposition.

```matlab
[trend, cycle] = quantecon.base.HpFilter(y);
[trend, cycle] = quantecon.base.HpFilter(y, 'Lambda', 1600);
```

#### VarHac

den Haan & Levin (1997) VARHAC long-run covariance estimator.

```matlab
Omega = quantecon.base.VarHac(data, 'MaxLag', 4, 'Criterion', 'BIC');
```

#### MatrixOps

Commutation, duplication, and elimination matrices (Magnus & Neudecker).

```matlab
K = quantecon.base.MatrixOps.Commutation(3, 4);
[D, Dinv] = quantecon.base.MatrixOps.Duplication(3);
L = quantecon.base.MatrixOps.Elimination(3);
```

---

### 2. `+panel` — Panel Data

#### FixedEffects

One-Way/Two-Way fixed effects with clustered standard errors.

```matlab
mdl = quantecon.panel.FixedEffects('TimeEffects', true, 'Cluster', 'TwoWay');
mdl = mdl.estimate(y, X, id, time);
```

#### ClusterReg

OLS with 1-way or 2-way clustering (Cameron, Gelbach & Miller 2011).

```matlab
results = quantecon.panel.ClusterReg(y, X, G);
```

#### DifferenceInDifferences

Classic 2x2 DiD estimator.

```matlab
mdl = quantecon.panel.DifferenceInDifferences();
res = mdl.estimate(y, treat, post, X, cluster);
```

#### `+panel.did` — Advanced DiD Estimators

| Class                   | Method                    | Reference                         |
| ----------------------- | ------------------------- | --------------------------------- |
| `CsEstimator`         | Group-Time ATT            | Callaway & Sant'Anna (2021)       |
| `SunAbraham`          | Interaction-Weighted      | Sun & Abraham (2021)              |
| `WooldridgeEstimator` | Two-Way Mundlak           | Wooldridge (2021)                 |
| `ImputationEstimator` | Counterfactual Imputation | Borusyak, Jaravel & Spiess (2024) |
| `SyntheticDid`        | Synthetic DiD             | Arkhangelsky et al. (2021)        |
| `BaconDecomposition`  | TWFE Decomposition        | Goodman-Bacon (2021)              |
| `Aggregator`          | ATT Aggregation           | Event Study / Group / Calendar    |

```matlab
% Callaway & Sant'Anna
mdl = quantecon.panel.did.CsEstimator();
mdl = mdl.estimate(y, treat, time, id);
agg = mdl.aggregate('EventStudy');

% Sun & Abraham
mdl = quantecon.panel.did.SunAbraham();
mdl = mdl.estimate(y, id, time, treat);

% Bacon Decomposition
res = quantecon.panel.did.BaconDecomposition.decompose(y, treat, time, id);
```

#### `+panel.var` — Panel Time Series

```matlab
% Panel VAR
mdl = quantecon.panel.var.PanelVar(Y, id, time, 'Lags', 2);

% Granger Causality
res = quantecon.panel.var.Granger.test(Y, id, time, 2);
```

---

### 3. `+timeseries` — Time Series

#### Garch / GarchK / GarchSk

Univariate volatility: standard GARCH, conditional kurtosis, conditional skewness.

```matlab
mdl = quantecon.timeseries.Garch(1, 1);
res = mdl.estimate(y);

results = quantecon.timeseries.GarchK(y, 1, 1);   % Brooks et al. (2005)
results = quantecon.timeseries.GarchSk(y, 1, 1);   % Leon et al. (2005)
```

#### GarchMidas

GARCH-MIDAS with long-run component (Engle, Ghysels & Sohn 2013).

```matlab
mdl = quantecon.timeseries.GarchMidas('Period', 22, 'NumLags', 10);
res = mdl.estimate(returns, rv_monthly);
```

#### Arima

Functional wrapper for MATLAB's native `arima`.

```matlab
res = quantecon.timeseries.Arima(y, 'p', 1, 'd', 0, 'q', 1);
```

#### Svar

Structural VAR with Cholesky / Long-Run (Blanchard-Quah) identification.

```matlab
mdl = quantecon.timeseries.Svar(Y, 2);
mdl = mdl.identify("chol");
irf_res = mdl.irf(20);
```

#### LocalProjection

Unified dispatcher for Local Projection impulse responses (Jorda 2005). Supports five methods via the `Method` option.

```matlab
% Standard LP (Jorda 2005)
res = quantecon.timeseries.LocalProjection(Y, horizon, lags);

% Lag-Augmented LP/VAR (Montiel Olea & Plagborg-Moller 2021)
res = quantecon.timeseries.LocalProjection(Y, horizon, lags, 'Method', 'lagaug');

% GLS-Efficient LP (Lusompa 2023)
res = quantecon.timeseries.LocalProjection(Y, horizon, lags, 'Method', 'gls');

% B-spline Smooth LP (Barnichon & Brownlees 2019)
res = quantecon.timeseries.LocalProjection(Y, horizon, lags, 'Method', 'smooth');

% Bayesian LP (Ferreira, Miranda-Agrippino & Ricco)
res = quantecon.timeseries.LocalProjection(Y, horizon, lags, 'Method', 'bayesian');
```

#### LpLagAug

Lag-Augmented Local Projections / VAR impulse responses (Montiel Olea & Plagborg-Moller 2021). Delta-method and bootstrap confidence intervals with Pope (1990) bias correction.

```matlab
res = quantecon.timeseries.LpLagAug(Y, 20, 4);
res = quantecon.timeseries.LpLagAug(Y, 20, 4, 'Method', 'VAR', 'Alpha', 0.1, 'NBoot', 500);
% res.IRF, res.CI_lo, res.CI_hi, res.BIC_lag
```

#### LpGls

GLS-Efficient Local Projections via Wold IRF correction (Lusompa 2023). Sandwich and block-wild bootstrap standard errors.

```matlab
res = quantecon.timeseries.LpGls(Y, 20, 4);
res = quantecon.timeseries.LpGls(Y, 20, 4, 'CIMethod', 'boot', 'NBoot', 500);
% res.IRF, res.CI_lo, res.CI_hi
```

#### LpSmooth

B-spline penalized smooth Local Projections (Barnichon & Brownlees 2019). LOO-CV for penalty selection.

```matlab
res = quantecon.timeseries.LpSmooth(Y, 20, 4);
res = quantecon.timeseries.LpSmooth(Y, 20, 4, 'NKnots', 5, 'SplineOrder', 3);
% res.IRF, res.CI_lo, res.CI_hi, res.Lambda
```

#### Blp

Bayesian Local Projections with Normal-Inverse-Wishart prior (Ferreira, Miranda-Agrippino & Ricco). Gibbs sampling posterior with optimized hyperparameters.

```matlab
res = quantecon.timeseries.Blp(Y, 20, 4);
res = quantecon.timeseries.Blp(Y, 20, 4, 'NDraws', 2000, 'NBurn', 500, 'Credibility', 0.90);
% res.IRF, res.CI_lo, res.CI_hi, res.Hyperparams
```

#### SvarIv

SVAR identification via external instruments (Montiel Olea, Stock & Watson 2021). Plug-in delta-method and MSW weak-IV robust Anderson-Rubin confidence intervals.

```matlab
res = quantecon.timeseries.SvarIv(Y, Z, 20, 4);
res = quantecon.timeseries.SvarIv(Y, Z, 20, 4, 'Alpha', 0.05, 'CIMethod', 'both');
% res.IRF, res.CI_dm_lo/hi, res.CI_ar_lo/hi, res.F_eff
```

#### SimBands

Simultaneous confidence bands for impulse response functions (Montiel Olea & Plagborg-Moller 2019). Methods: sup-t, Bonferroni, Sidak, projection, pointwise, calibrated.

```matlab
bands = quantecon.timeseries.SimBands(IRF_draws);
bands = quantecon.timeseries.SimBands(IRF_draws, 'Method', 'sup-t', 'Alpha', 0.10);
% bands.lo, bands.hi, bands.Method
```

#### MarkovSwitching

Markov Regime-Switching model with EM estimation.

```matlab
results = quantecon.timeseries.MarkovSwitching(y, x);
```

#### MarkovSim

Empirical Markov Chain estimation and simulation.

```matlab
[sim, P] = quantecon.timeseries.MarkovSim(data, 1000);
```

#### Fcvar

Fractionally Cointegrated VAR (FCVAR).

```matlab
results = quantecon.timeseries.Fcvar(x, k, r, [d, b]);
```

#### Midas (timeseries)

Mixed-frequency data sampling regression.

```matlab
results = quantecon.timeseries.Midas(y, X_hf, 'NumLags', 12);
```

#### `+timeseries.fou` — Fractional Ornstein-Uhlenbeck

```matlab
res = quantecon.timeseries.fou.AwmlFit(x, "K", 50, "Delta", 1/252);
y_sim = quantecon.timeseries.fou.Simulate(1.0, 0.0, 0.5, 0.7, 5, 1/252, 8, 0);
```

#### `+timeseries.longmemory` — Long Memory / ARFIMA

```matlab
lw = quantecon.timeseries.longmemory.LocalWhittle(y, 0.2, 50, "none");
[d0, a0] = quantecon.timeseries.longmemory.WhittleInitial(y, (-0.4:0.01:0.4)', (0.1:0.05:0.9)');
fcst = quantecon.timeseries.longmemory.ForecastArfima(y, [0.2; 0.8; 1.0], 5, mean(y));
```

#### HurstExponent

Hurst exponent via R/S analysis, DFA, and wavelet methods.

```matlab
mdl = quantecon.timeseries.HurstExponent(x);
fprintf('R/S = %.3f, DFA = %.3f\n', mdl.RS, mdl.DFA);
```

#### Dekf

Dual Extended Kalman Filter for joint state and parameter estimation.

```matlab
mdl = quantecon.timeseries.Dekf(y, 'StateEq', @f, 'ObsEq', @h);
res = mdl.estimate();
```

#### `+timeseries.disagg` — Temporal Disaggregation

Convert low-frequency to high-frequency data (Chow-Lin, Denton, Fernandez, Litterman).

```matlab
res = quantecon.timeseries.disagg.ChowLin(Y_low, X_high, s, 'Type', 'SSR');
res = quantecon.timeseries.disagg.Denton(Y_low, X_high, s);
res = quantecon.timeseries.disagg.Litterman(Y_low, X_high, s);
res = quantecon.timeseries.disagg.Fernandez(Y_low, X_high, s);
```

---

### 4. `+bayes` — Bayesian Econometrics

#### Bvar

Bayesian VAR with Minnesota / Conjugate / Diffuse priors.

```matlab
mdl = quantecon.bayes.Bvar(2, "minnesota");
res = mdl.estimate(Y);
```

#### BvarSsvs

BVAR with Stochastic Search Variable Selection.

```matlab
res = quantecon.bayes.BvarSsvs(Y, 2, 'tau0', 0.1, 'tau1', 10);
```

#### BvarSv

BVAR with Stochastic Volatility (Gibbs Sampling).

```matlab
results = quantecon.bayes.BvarSv(Y, 1, 'nsamp', 1000);
```

#### TvpVar

Time-Varying Parameter VAR with Stochastic Volatility.

```matlab
mdl = quantecon.bayes.TvpVar(Y, 1, 'nsamp', 1000);
```

#### Favar

Factor-Augmented VAR (Bernanke, Boivin & Eliasz 2005).

```matlab
mdl = quantecon.bayes.Favar(3, 1, "twostep");
res = mdl.estimate(Y_large, Y_var);
```

#### FactorVar

Two-step Factor-Augmented Bayesian VAR (BBE 2005) — simplified wrapper with PCA factor extraction and Minnesota prior.

```matlab
results = quantecon.bayes.FactorVar(XY, [], 3, 1);
```

#### Mfbvar

Mixed-Frequency BVAR (Schorfheide & Song 2015).

```matlab
mdl = quantecon.bayes.Mfbvar(1);
res = mdl.estimate(Ym, Yq);
```

#### PanelVar

Bayesian Panel VAR for multi-unit data.

```matlab
mdl = quantecon.bayes.PanelVar(Y_panel, 1, 'FixedEffects', true);
```

#### SignId

Sign Restriction Engine for structural shocks (Rubio-Ramirez et al. 2010).

```matlab
[irfs, Q] = quantecon.bayes.SignId.identify(Coeffs, Sigma, p, 20, R, 500);
```

#### BvarAnalysis

Post-estimation: IRF, FEVD, Historical Decomposition.

```matlab
irfs = quantecon.bayes.BvarAnalysis.irf(coeffs, Sigma, p, 20, "chol");
fevd = quantecon.bayes.BvarAnalysis.fevd(coeffs, Sigma, p, 20);
hd   = quantecon.bayes.BvarAnalysis.hd(coeffs, Sigma, p, Y);
```

#### BEAR Integration

Bridge to BEAR Toolbox for full BVAR/TVP/SV/MFVAR pipelines.

```matlab
opts = quantecon.bayes.BearSettings(2, "ExcelFile", "my_data.xlsx");
quantecon.bayes.BearRun(opts);
```

#### MhSampler

Metropolis-Hastings MCMC sampler with adaptive tuning.

```matlab
mdl = quantecon.bayes.MhSampler(@logpost, theta0, 'NSamp', 5000, 'Proposal', 'normal');
res = mdl.sample();
```

#### HpdInterval

Highest Posterior Density credible intervals from MCMC draws.

```matlab
bands = quantecon.bayes.HpdInterval(draws, 'Prob', 0.9);
bands = quantecon.bayes.HpdInterval(draws, 'Prob', 0.9, 'Shortest', true);
```

#### GaussHermite

Gauss-Hermite quadrature nodes and weights (Golub-Welsch algorithm).

```matlab
[x, w] = quantecon.bayes.GaussHermite(5);
```

---

### 5. `+tests` — Statistical Tests & Bubble Detection

#### Adf

Augmented Dickey-Fuller with AIC/BIC lag selection.

```matlab
tstat = quantecon.tests.Adf(y, 0, 2);       % Fixed lag = 2
tstat = quantecon.tests.Adf(y, 2, 4);       % BIC-selected (max 4)
```

#### AdfRes

Restricted ADF regression under the unit-root null. Provides null-hypothesis residuals for the wild bootstrap procedure (PsyBoot).

```matlab
[beta, eps, lag] = quantecon.tests.AdfRes(y, 2, 4);
```

#### Psy / PsyCv / PsyBoot

Phillips-Shi-Yu (2015) BSADF bubble detection.

```matlab
bsadf = quantecon.tests.Psy(PriceSeries, 20, 2, 4);
gsadf = max(bsadf);

% Monte Carlo critical values
Q = quantecon.tests.PsyCv(T, swindow0, 2, 4, NumSim=2000);

% Wild bootstrap CVs (Shi & Phillips 2023)
Q = quantecon.tests.PsyBoot(y, swindow0, 2, 4, Horizon=4);
```

#### Ivx / IvxAr

IVX predictive regression for persistent regressors (Kostakis et al. 2015).

```matlab
res = quantecon.tests.Ivx(y, X);
res = quantecon.tests.IvxAr(y, X, AR=1, MaxArLag=5);
```

#### DateStamp

Date-stamp bubble episodes from binary indicator.

```matlab
episodes = quantecon.tests.DateStamp(bsadf > cv95, dates);
```

#### Dfgls

Elliott-Rothenberg-Stock DF-GLS unit root test with AIC/BIC lag selection.

```matlab
res = quantecon.tests.Dfgls(y, 'MaxLag', 10, 'LagSelect', 'bic', 'Trend', 'c');
fprintf('stat = %.3f, cv5%% = %.3f\n', res.stat, res.cv5);
```

#### Mcs

Model Confidence Set (Hansen, Lunde & Nason 2011).

```matlab
res = quantecon.tests.Mcs(LossMatrix, 'Alpha', 0.10, 'NumBoot', 5000);
disp(res.Included)   % models in superior set
```

#### Bsds

White's Reality Check & Hansen's Superior Predictive Ability (SPA) test.

```matlab
[c, u, l] = quantecon.tests.Bsds(benchLoss, modelLosses, 'B', 1000, 'BlockLen', 12);
% c = Hansen SPA p-value, u = White RC p-value, l = lower bound
% For "goods" (returns): pass -bench, -models
```

#### BreuschPagan

Breusch-Pagan / Koenker test for heteroskedasticity.

```matlab
res = quantecon.tests.BreuschPagan(y, X, 'Robust', true);
fprintf('BP stat = %.3f, pval = %.4f\n', res.stat, res.pval);
```

#### MultiLbq

Multivariate Ljung-Box (Hosking 1980) portmanteau test.

```matlab
res = quantecon.tests.MultiLbq(Residuals, 'MaxLag', 12);
```

#### TvGranger

Time-Varying Granger Causality with rolling/recursive/bootstrap methods.

```matlab
mdl = quantecon.tests.TvGranger(Y, 'Lags', 2, 'WindowType', 'rolling', 'WindowSize', 60);
res = mdl.estimate();
```

#### HomTest

Homoscedasticity test suite: White, Breusch-Pagan, Goldfeld-Quandt.

```matlab
res = quantecon.tests.HomTest(y, X, 'Test', 'white');
```

---

### 6. `+multivariate` — Multivariate Analysis

#### Dcc

DCC-GARCH (Engle 2002), 2-stage estimation.

```matlab
mdl = quantecon.multivariate.Dcc('M', 1, 'N', 1);
res = mdl.estimate(Y);
```

#### Bekk

BEKK-GARCH(1,1) with positive-definite covariance.

```matlab
results = quantecon.multivariate.Bekk(Y, 'Type', 'Scalar');
```

#### DccMidas

DCC-MIDAS model (Colacito, Engle & Ghysels 2011).

```matlab
mdl = quantecon.multivariate.DccMidas('Period', 22);
res = mdl.estimate(Y);
```

#### CopulaFit

Static copulas: Gaussian, t, Clayton, SJC.

```matlab
u = quantecon.base.EmpiricalCdf(Y);
res = quantecon.multivariate.CopulaFit(u, 't');
```

#### ReducedRankReg

Reduced Rank Regression (Anderson 1951).

```matlab
res = quantecon.multivariate.ReducedRankReg(y, x, 2);
```

---

### 7. `+finance` — Financial Econometrics

#### Connectedness

Advanced framework replicating the R `ConnectednessApproach` package.
Supports Diebold-Yilmaz (2009), Barunik-Krehlik (2018), Quantile VAR (QVAR), Elastic Net VAR, and Joint/Extended Joint Decompositions (Lastrapes & Wiesen, 2021).

```matlab
% 1. Model Estimation (e.g. Quantile VAR at Tau=0.50)
cn = quantecon.finance.Connectedness(data);
cn.estimate('Method', 'qvar', 'Lags', 1, 'Tau', 0.5);

% 2. Forecast Error Variance Decomposition (e.g. Joint Decomposition)
cn.decompose('Horizon', 10, 'DecompType', 'joint');
total = cn.Indices.Total;

% 3. Grammar-of-Graphics Visualization
quantecon.vis.ConnectednessVis.plotTCI(cn);
quantecon.vis.ConnectednessVis.plotNET(cn);
quantecon.vis.ConnectednessVis.plotDirectional(cn, 'FROM');
quantecon.vis.ConnectednessVis.plotNetwork(cn);
```

#### DynamicNetworks

Frequency-domain network connectedness (Barunik & Ellington 2020).

```matlab
res = quantecon.finance.DynamicNetworks(data, "NumSim", 100, "Lags", 1);
```

#### TechnicalIndicators

MA, Momentum, OBV.

```matlab
res = quantecon.finance.TechnicalIndicators(price, volume);
```

#### AssetIndicators

Amihud Illiquidity, Idiosyncratic Volatility, etc.

```matlab
val = quantecon.finance.AssetIndicators("amihud", ret, dollar_vol);
val = quantecon.finance.AssetIndicators("ivol", ret, [mkt, smb, hml]);
```

#### HistVol

Historical volatility estimators: Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang.

```matlab
mdl = quantecon.finance.HistVol(prices, 'Window', 21, 'Method', 'yang-zhang');
vol = mdl.estimate();
```

#### HestonPrice

Heston (1993) stochastic volatility option pricing via Fourier inversion.

```matlab
price = quantecon.finance.HestonPrice(S0, K, r, T, v0, kappa, theta, sigma, rho);
```

#### CommodityFutures

Multi-factor commodity futures pricing via Kalman filter MLE (Ballestra & Tezza, *Quantitative Finance*). Estimates a 4-factor state-space model where the latent state vector is [log spot price, convenience yield, interest rate, stochastic volatility]. Futures prices follow a quasi-affine representation with coefficients computed via 4th-order Runge-Kutta.

Supports two estimation modes:

- **Mode A (commodity-only)**: all 4 factors latent, inferred from futures prices alone
- **Mode B (with rates)**: interest rate observed from bond yields, appended as an extra observation equation `r_obs = [0,0,1,0] * x_t + eta_t`

```matlab
% Mode A: commodity-only (all factors latent)
result = quantecon.finance.CommodityFutures(LogFutures, [2 4 6 8 10]/12);

% Mode B: with observed interest rates (better r identification)
result = quantecon.finance.CommodityFutures(LogFutures, Maturities, ...
    'Rates', r_observed);

% Full options
result = quantecon.finance.CommodityFutures(LogFutures, Maturities, ...
    'Rates', r_obs, 'TimeStep', 1/252, 'NRunge', 2000, 'NStarts', 200);

% Access results
fprintf('LogLik = %.2f, AIC = %.2f, BIC = %.2f\n', ...
    result.LogLik, result.AIC, result.BIC);
disp(result.ParamTable);        % named parameter estimates
plot(result.States(:,1));       % filtered log spot price
title('Filtered Log Spot Price');
```

---

### 8. `+risk` — Systemic Risk

#### Covar

Conditional Value-at-Risk (Adrian & Brunnermeier 2016).

```matlab
mdl = quantecon.risk.Covar(0.05);
res = mdl.estimate(SystemRet, InstRet, StateVars);
```

#### Mes

Marginal Expected Shortfall (Acharya et al. 2017).

```matlab
mdl = quantecon.risk.Mes(0.05);
res = mdl.estimate(SystemRet, InstRet);
```

#### Srisk

SRISK Capital Shortfall (Acharya, Engle & Richardson 2012).

```matlab
mdl = quantecon.risk.Srisk(0.08);
res = mdl.estimate(SystemRet, InstRet, Liabilities, Equity);
```

---

### 9. `+ml` — Machine Learning

#### CausalForest

Heterogeneous Treatment Effects with multiple estimators: T-learner, X-learner, Honest (Wager & Athey 2018), DR-Learner (AIPW), and Quantile Treatment Effects (QTE).

```matlab
% T-learner (basic)
cf = quantecon.ml.CausalForest('Estimator', 'tlearner', 'NumTrees', 100);
cate = cf.estimate(X, Y, W);

% Honest causal forest (Wager & Athey 2018)
cf = quantecon.ml.CausalForest('Estimator', 'honest', 'NumTrees', 200);
res = cf.estimate(X, Y, W);

% DR-Learner with cross-fitted nuisance
cf = quantecon.ml.CausalForest('Estimator', 'dr', 'NumFolds', 5);
res = cf.estimate(X, Y, W);

% Quantile Treatment Effects (conditional & unconditional)
cf = quantecon.ml.CausalForest('Estimator', 'tlearner');
cf.estimate(X, Y, W);
qte = cf.qte(X, Y, W, 'Quantiles', [0.1, 0.25, 0.5, 0.75, 0.9]);
```

#### Ddml

Double/Debiased Machine Learning with cluster-robust standard errors.

```matlab
ddml = quantecon.ml.Ddml('ModelY', 'Lasso', 'ModelT', 'Lasso');
res = ddml.estimate(Y, T, X);

% With 1-way or 2-way clustering
res = ddml.estimate(Y, T, X, 'ClusterID', G);
res = ddml.estimate(Y, T, X, 'ClusterID', [G1, G2]);  % two-way
```

---

### 10. `+causal` — Causal Inference

#### Rdd

Regression Discontinuity Design (Sharp/Fuzzy) with IK/CCT bandwidth.

```matlab
rdd = quantecon.causal.Rdd('Cutoff', 0, 'Kernel', 'triangular', 'BandwidthMethod', 'ik');
res = rdd.estimate(RunningVar, Outcome);
res = rdd.estimate(RunningVar, Outcome, Treatment);  % Fuzzy
```

#### SemFit

Structural Equation Modeling with bootstrap diagnostics.

```matlab
mdl = quantecon.causal.SemFit(S, Model_Spec);
res = mdl.estimate('Bootstrap', true, 'nBoot', 500);
```

#### LpDid

Local Projection Difference-in-Differences (Dube, Girardi, Jorda & Taylor 2025). Clean control condition, variance-weighted (VW) and regression-weighted (RW) estimators, clustered standard errors.

```matlab
res = quantecon.causal.LpDid(y, treat, time, id, 8);
res = quantecon.causal.LpDid(y, treat, time, id, 8, 'Weighting', 'rw', 'Alpha', 0.10);
% res.IRF, res.SE, res.CI_lo, res.CI_hi (horizon x 1 each)
```

#### Psm

Propensity Score Matching for causal inference with multiple PS models, matching algorithms, Abadie-Imbens bias correction and standard errors, and PSM-DID integration.

PS models: `logit`, `probit`, `lasso` (L1-penalized), `rf` (random forest).
Matching: `nearest` (1:k NN, with/without replacement), `kernel`, `radius`.
Treatment effects: ATT, ATE, ATNT.

```matlab
% Stata-compatible default: without replacement + Abadie-Imbens bias correction
psm = quantecon.causal.Psm('WithReplacement', false, 'BiasCorrection', true);
res = psm.estimate(Y, W, X);
fprintf('ATT = %.4f (SE = %.4f)\n', res.ATT, res.SE_ATT);

% NN(1:5) matching with ML propensity score
psm = quantecon.causal.Psm('PSModel', 'rf', 'NumNeighbors', 5);
res = psm.estimate(Y, W, X);

% Kernel matching
psm = quantecon.causal.Psm('MatchMethod', 'kernel', 'Bandwidth', 0.06);
res = psm.estimate(Y, W, X);

% PSM-DID (Heckman, Ichimura & Todd 1997)
res = psm.estimateDid(Y_pre, Y_post, W, X);
fprintf('PSM-DID ATT = %.4f (SE = %.4f)\n', res.ATT_DID, res.SE_DID);

% Balance diagnostics
disp(res.Balance)
```

---

### 11. `+factor` — Asset Pricing Factors

#### Ipca

Instrumented PCA (Kelly, Pruitt & Su 2019).

```matlab
mdl = quantecon.factor.Ipca(K);
mdl = mdl.estimate(Returns, Characteristics);
[Gamma, Factors] = mdl.infer();
```

#### Spca

Scaled PCA (Huang et al. 2022).

```matlab
mdl = quantecon.factor.Spca(K);
res = mdl.estimate(Target, Predictors);
```

#### Dfm

Dynamic Factor Model via EM-Kalman Filter/Smoother (Miranda-Agrippino 2013). PCA initialization, VAR(p) factor dynamics.

```matlab
res = quantecon.factor.Dfm(X, 3);
res = quantecon.factor.Dfm(X, 3, 'Lags', 2, 'MaxIter', 200, 'Tol', 1e-6);
% res.F (T x r factors), res.C (loadings), res.A (transition), res.LogLik
```

#### DfmBlock

Block-Restricted Dynamic Factor Model (Banbura & Modugno 2010). Supports block structure constraints, idiosyncratic AR dynamics, and missing data (NaN).

```matlab
% blockid: N x 1 integer vector mapping each variable to its block
res = quantecon.factor.DfmBlock(X, [1;1;2;2;3;3], 'RPerBlock', [1 1 1]);
res = quantecon.factor.DfmBlock(X, blockid, 'RPerBlock', r_vec, 'IdioLags', 1, 'MaxIter', 200);
% res.F (factors), res.C (loadings), res.A (transition), res.R (idio var), res.LogLik
```

---

### 12. `+robust` — Robust Statistics (FSDA)

#### RobustReg

Least Trimmed Squares (LTS) regression.

```matlab
results = quantecon.robust.RobustReg(y, X, 'nsamp', 500);
```

#### RobustCov

Minimum Covariance Determinant (MCD).

```matlab
results = quantecon.robust.RobustCov(Y, 'H', floor(T*0.75));
```

#### RobustPca

Robust PCA based on MCD covariance.

```matlab
results = quantecon.robust.RobustPca(Y, 3);
```

#### RobustSearch

Forward Search for outlier detection (FSDA-inspired).

```matlab
results = quantecon.robust.RobustSearch(Y);
```

#### TrimmedCluster

Robust clustering with trimming and eigenvalue restrictions (Garcia-Escudero et al. 2008).

```matlab
mdl = quantecon.robust.TrimmedCluster(3, 0.1, 10);
res = mdl.fit(Y);
```

---

### 13. `+midas` — MIDAS Regression

Mixed Data Sampling with analytic derivatives and Beta/ExpAlmon polynomials.

```matlab
results = quantecon.midas.Midas(y, X_hf, 'Polynomial', 'Beta');
res = quantecon.midas.estimate_midas(y, X_hf, theta0);
```

---

### 14. `+forecasting` — Forecast Evaluation

#### BmaForecast

Bayesian Model Averaging for forecast combination.

```matlab
forecast = quantecon.forecasting.BmaForecast(forecast_cell, weights);
```

#### MurphyDiagram

Visual comparison of two forecasts (Ehm et al. 2016).

```matlab
res = quantecon.forecasting.MurphyDiagram(y, f1, f2);
```

---

### 15. `+models` — Specialized Models

#### QbllVar

Quasi-Bayesian Local Likelihood VAR (Petrova 2019).

```matlab
mdl = quantecon.models.QbllVar(Y, 'Lags', 2, 'Shrinkage', 0.05);
mdl = mdl.estimate();
irf = mdl.irf(10);
```

---

### 16. `+sem` — Structural Equation Modeling

#### PlsEstimator

PLS Path Modeling (Wold 1982).

```matlab
mdl = quantecon.sem.PlsEstimator(DP, DB);
mdl = mdl.estimate(MV);
```

---

### 17. `+optimization` — Optimization & Numerical Methods

#### Simplex

Constrained least-squares simplex solver.

```matlab
w = quantecon.optimization.Simplex.solve(X, target, zeta);
```

#### CsminWel

Chris Sims' quasi-Newton minimizer with BFGS Hessian update. Robust to cliff edges and bad gradients.

```matlab
[fhat, xhat, ghat, Hhat] = quantecon.optimization.CsminWel(@obj, x0, eye(n), [], 1e-8, 200);
```

#### NumGrad

Forward-difference numerical gradient.

```matlab
[g, badg] = quantecon.optimization.NumGrad(@fcn, x0);
```

---

### 18. `+utils` — Utilities

#### VarUtils

Shared utilities for VAR and Connectedness models: design matrix setup, Gaussian kernel weights, Generalized IRF.

```matlab
[X, Y] = quantecon.utils.VarUtils.var_setup(Data, p);
w = quantecon.utils.VarUtils.norm_kernel(T, H);
ir = quantecon.utils.VarUtils.compute_girf(A_comp, Sigma, N, L, H);
```

---

### 19. `+vis` — Visualization

#### Gramm

Grammar of Graphics for MATLAB. From [`piermorel/gramm`](https://github.com/piermorel/gramm).

```matlab
g = quantecon.vis.Gramm('x', x, 'y', y, 'color', categories);
g.geom_point();
g.geom_line();
g.draw();
```

#### notBoxPlot

Alternative to box plots showing raw data with SEM/SD patches. From raacampbell.

```matlab
quantecon.vis.notBoxPlot(y, x);
```

#### exportFig

High-quality figure export (PNG, PDF, EPS) with auto-cropping.

```matlab
quantecon.vis.exportFig(gcf, 'my_figure.pdf');
```

---

### 20. `+dsge` — DSGE Model Toolkit

#### Gensys

Sims (2002) QZ-based solver for linear rational expectations models.

```matlab
[G1, C, impact, ~, ~, ~, ~, eu] = quantecon.dsge.Gensys(g0, g1, c, psi, pi);
```

#### PertSolve

First-order perturbation solution via generalized Schur (Klein 2000).

```matlab
[SOL, stab] = quantecon.dsge.PertSolve(A, B, nx);
% SOL.F = decision rule, SOL.P = law of motion
```

#### ProjSolve

Projection method solver using Chebyshev collocation (Judd 1998).

```matlab
[POL, GRID, exitflag] = quantecon.dsge.ProjSolve(@residual, lb, ub, Y0, 'Order', 5);
```

#### Tauchen

Tauchen (1986) AR(1) discretization into finite Markov chain.

```matlab
[grid, P] = quantecon.dsge.Tauchen(rho, sigma, 'N', 7, 'M', 3);
```

#### Rouwenhorst

Rouwenhorst (1995) AR(1) discretization (better for high persistence).

```matlab
[grid, P] = quantecon.dsge.Rouwenhorst(rho, sigma, 'N', 5);
```

#### KalmanFilter

Linear Kalman filter with NaN (missing data) support.

```matlab
res = quantecon.dsge.KalmanFilter(data, F, H, 'Q', Q, 'R', R);
% res.LogLik, res.Filt, res.Pred
```

#### KalmanSmoother

Koopman (1993) disturbance smoother for state-space models.

```matlab
res = quantecon.dsge.KalmanSmoother(data, F, H, R_shock, Q_noise);
% res.SmoothedStates, res.SmoothedShocks
```

#### PriorDens

Log prior density for Bayesian DSGE estimation (Beta/Gamma/Normal/InvGamma/Uniform).

```matlab
lp = quantecon.dsge.PriorDens(para, pmean, pstdd, pshape);
```

#### DiscLyap

Discrete Lyapunov equation solver via doubling algorithm: X = A*X*A' + C.

```matlab
X = quantecon.dsge.DiscLyap(A, C);
```

---

## Testing

```matlab
% Full verification suite
tests.test_enrichment()

% Visualization verification
verify_vis_refinement
verify_vis_export

% Legacy benchmark suite
tests.run_benchmarks
```

### Cross-Verification (Stata & Python)

Validated against Stata (`reghdfe`, `xtreg`, `teffects psmatch`) and Python (`linearmodels`, `statsmodels`, `arch`, `sklearn`).

| Test             | MATLAB vs Stata | MATLAB vs Python |
| ---------------- | --------------- | ---------------- |
| OLS coefficients | < 1e-5          | < 1e-5           |
| Panel FE         | < 1e-6          | < 1e-5           |
| GARCH(1,1)       | —              | < 0.02           |
| ADF statistic    | matched         | matched          |
| PSM ATT (NN1)    | < 0.04          | < 1e-4           |
| PSM ATT (NN5+BC) | < 0.02          | —               |
| PSM-DID ATT      | < 0.04          | < 0.05           |

```matlab
run('verification/orchestrate_verification.m')
```
