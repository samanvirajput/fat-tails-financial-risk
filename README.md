# Fat Tails, Volatility Clustering & Risk Underestimation in Stock Returns

> Rigorous comparative statistical analysis of AAPL and TSLA daily log returns (2015–2024), demonstrating systematic failure of Gaussian VaR models. arXiv submission forthcoming — q-fin.ST / stat.AP.

## Paper
**"Beyond Normality: A Comparative Statistical Analysis of Fat Tails, Volatility Clustering, and Risk Underestimation in Stock Market Returns"**
*Samanvi Rajput — School of Computer Science and Engineering, VIT Vellore*

📄 arXiv preprint: forthcoming (q-fin.ST / stat.AP)

## Key Findings

| Finding | AAPL | TSLA |
|---|---|---|
| Excess kurtosis | 4.62 | 5.74 |
| Student-t degrees of freedom (ν̂) | 4.73 | 3.62 |
| AIC improvement over Gaussian | 496 | 588 |
| Normal VaR underestimation @ 99.9% | $21,700 | $104,200 |
| Volatility clustering (Ljung-Box Q(10)) | 214.3 | 341.2 |

All four hypotheses confirmed at α = 0.05:
- **H1**: Both assets exhibit statistically significant leptokurtosis and negative skewness
- **H2**: Student-t provides decisively superior fit (ΔAIC > 400 for both assets)
- **H3**: Volatility clustering confirmed — squared return autocorrelation significant at all tested lags
- **H4**: Gaussian VaR underestimates empirical 99.9%-VaR by up to $104,200 per $1M exposure (TSLA)

## Methodology
- Moment analysis (skewness, excess kurtosis)
- Shapiro-Wilk + Kolmogorov-Smirnov normality tests
- Maximum likelihood estimation: Normal vs Student-t
- AIC-based model selection
- Ljung-Box test on squared returns (ARCH effects)
- Parametric + empirical + Student-t Value-at-Risk

## How to Run
```bash
pip install -r requirements.txt
python fat_tails_analysis.py
```
Outputs 6 publication-quality figures to the current directory.

## Data
Daily adjusted closing prices for AAPL and TSLA (2015–2024) fetched live via `yfinance`. No static data files required.

## Stack
`Python` `NumPy` `SciPy` `pandas` `statsmodels` `matplotlib` `yfinance`
