# Quantitative Sports Betting Analysis System

A computational framework for analyzing the U.S. sports betting market through the lens of quantitative finance. The project implements Elo-based prediction, Monte Carlo simulation, expected value analysis, Kelly Criterion position sizing, and full strategy backtesting — demonstrating that the mathematical toolkit for sports betting is structurally identical to that of quantitative trading.

---

## Overview

This project treats the sports betting market as a financial market and applies rigorous quantitative methods to answer a central question: **can a mathematical model systematically identify and exploit mispricings in sports betting odds?**

The system processes 720 simulated NBA games across 16 teams, producing:
- **Elo ratings** with home-court advantage and margin-of-victory adjustment
- **10,000-path Monte Carlo** season simulations with playoff probability distributions
- **Walk-forward backtesting** of a +EV betting strategy using Kelly Criterion sizing
- **Model calibration** via Brier score (0.2312) and log loss (0.6550)
- **Market microstructure analysis** including vig decomposition, CLV, and price discovery
- **18 publication-quality figures** across 7 core and 11 extended analyses

The backtest produces a negative ROI (−5.81%), which is itself a key finding: a basic Elo model, while well-calibrated, cannot overcome the sportsbook's vigorish — the sports-betting analogue of the efficient market hypothesis.

## Key Results

| Metric | Value |
|---|---|
| Games Analyzed | 720 |
| Bets Placed | 531 |
| Win Rate | 48.0% |
| ROI | −5.81% |
| Max Drawdown | 95.1% |
| Brier Score | 0.2312 (baseline: 0.2500) |
| Log Loss | 0.6550 (baseline: 0.6931) |
| Average Market Vig | 4.53% |

## Architecture

```
sports_betting_analysis.py
│
├── OddsConverter          — American/decimal/fractional odds & vig decomposition
├── EVCalculator           — Expected value & edge identification
├── EloSystem              — Team ratings with home-court & MOV adjustment
├── MonteCarloSimulator    — N-path season simulation & playoff probabilities
├── ParlayAnalyzer         — Compound probability decay & house edge growth
├── MarketEfficiency       — Calibration testing, Brier score, log loss
├── KellyCriterion         — Optimal & fractional position sizing
├── DataGenerator          — Synthetic NBA season with realistic noise & vig
├── Backtester             — Walk-forward +EV strategy with risk metrics
└── Visualizer             — 18-figure production engine
```

All mathematical models are implemented from first principles using only **NumPy** and **Matplotlib** — no black-box libraries.

## Figures

The system generates 18 figures covering:

| # | Figure | Description |
|---|---|---|
| 1 | Parlay Decay | Exponential win probability collapse & compounding house edge |
| 2 | Elo Ratings | Post-season team rating distribution (16 teams) |
| 3 | Monte Carlo | 10K-sim win total distribution with normal approximation |
| 4 | Backtest (4 panels) | Bankroll trajectory, cumulative P&L, edge distribution, win rate by quartile |
| 5 | Calibration | Predicted probability vs observed win rate |
| 6 | Kelly Criterion | Bankroll paths at full/half/quarter/tenth Kelly |
| 7 | Vig Analysis | Market overround distribution & EV distribution |
| 8 | Sensitivity Heatmap | ROI across edge threshold × Kelly fraction parameter space |
| 9 | Poisson Model (3 panels) | Score distributions, O/U pricing, win probability matrix |
| 10 | Risk of Ruin (2 panels) | Ruin probability curves & drawdown distribution |
| 11 | CLV Analysis (2 panels) | Closing line value scatter & distribution |
| 12 | Sharpe Comparison | Cross-asset risk-adjusted return benchmarking |
| 13 | Information Decay | Edge half-life by information source |
| 14 | Price Discovery (2 panels) | Line movement simulation & volume-impact correlation |
| 15 | Bayesian Updating (2 panels) | Live in-game win probability & score margin evolution |
| 16 | Portfolio Theory (3 panels) | Correlation matrix, efficient frontier, diversification curve |
| 17 | Signal Analysis (2 panels) | Feature half-life decay & mutual information ranking |
| 18 | Market Maker (2 panels) | Sportsbook P&L simulation & hold distribution |

## Installation

```bash
git clone https://github.com/yourusername/quantitative-sports-betting.git
cd quantitative-sports-betting
pip install numpy matplotlib
```

No other dependencies required.

## Usage

```bash
python sports_betting_analysis.py
```

Output is written to `./project_outputs/`:
```
project_outputs/
├── fig1_parlay_decay.png
├── fig2_elo_ratings.png
├── fig3_monte_carlo.png
├── fig4_backtest.png
├── fig5_calibration.png
├── fig6_kelly.png
├── fig7_vig.png
└── summary.json
```

The full pipeline (Elo initialization → data generation → parlay analysis → season processing → Monte Carlo → backtest → calibration → visualization) runs in under 10 seconds.

## Mathematics

The report (`Quantitative_Sports_Betting_Report.pdf`) provides full mathematical derivations for all models. Key equations:

**Elo Expected Score:**

$$E(A) = \frac{1}{1 + 10^{(R_B - R_A^*)/400}}$$

**Kelly Criterion (optimal bet fraction):**

$$f^* = \frac{bp - q}{b}$$

where $b$ is net decimal odds, $p$ is true win probability, $q = 1-p$.

**Parlay House Edge (k legs, per-leg vig v):**

$$\text{HE}_k = (1 + v)^k - 1$$

**Bayesian In-Game Win Probability:**

$$P(\text{win} \mid m_t, \tau) \approx \Phi\!\left(\frac{m_t}{\sigma\sqrt{\tau}}\right)$$

**Brier Score:**

$$\text{BS} = \frac{1}{n}\sum_{i=1}^{n}(p_i - o_i)^2$$

## Topics Covered

- **Market Microstructure** — Price discovery, sharp vs public money, Kyle's lambda
- **Bayesian Inference** — Real-time posterior updating during live games
- **Portfolio Theory** — Markowitz optimization applied to a book of simultaneous bets
- **Signal Analysis** — Feature half-life estimation, mutual information ranking
- **Market Making** — Sportsbook P&L simulation, inventory risk, hold percentage
- **Risk Management** — Drawdown analysis, risk of ruin, Kelly fraction sensitivity
- **Arbitrage Detection** — Cross-platform guaranteed profit framework
- **Closing Line Value** — Industry-standard performance evaluation metric

## Report

The accompanying 57-page LaTeX report follows a rigorous structure:

1. **Executive Summary**
2. **Statement of the Problem** — Five sub-problems with mathematical formulation
3. **Description of the Mathematics** — Full derivations for all models
4. **Description of the Algorithm** — Pseudocode, complexity analysis, implementation details
5. **Description of the Testing** — Unit validation, calibration, backtest methodology
6. **Results and Discussion** — 18 figures with detailed quantitative analysis
7. **Conclusion** — Six principal findings including the market efficiency result
8. **References** — 19 academic and industry sources

## Reproducibility

All results are deterministic. The random seed is set to `2026`:
```python
np.random.seed(2026)
```

Running the script on any machine with NumPy installed will produce identical figures and summary statistics.

## License

MIT

## References

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Thorp, E. O. (2017). *A Man for All Markets*. Random House.
- Silver, N. (2014). Introducing NFL Elo ratings. *FiveThirtyEight*.
- Dixon, M. J. & Coles, S. G. (1997). Modelling association football scores. *JRSS Series C*, 46(2), 265–280.
- Stern, H. S. (1994). A Brownian motion model for sports scores. *JASA*, 89(427), 1128–1134.
