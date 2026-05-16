# Professional Options Entry Research: Methodology, Academic Basis, and Tool Development Guide

**Purpose:** This document synthesizes academic literature, practitioner research, and professional trading methodology covering how institutional options traders and analysts plan entries and conduct background research before selecting positions. It is intended to guide feature development for the custom options analyzer tool (`custom_model.py`).

**Scope:** Ten core topic areas are covered, each with academic grounding, practitioner standards, and quantitative thresholds. A final section maps findings to the current tool's capabilities and identifies priority gaps.

---

## Table of Contents

1. Implied Volatility Rank (IVR) and IV Percentile
2. IV vs. Realized Volatility (HV) — The Volatility Risk Premium
3. Term Structure of Volatility / VIX Term Structure
4. Greeks-Based Entry Criteria
5. Expected Value / Edge-Based Entry
6. Liquidity Filters
7. Earnings / Event-Driven Analysis
8. Sector / Macro Context
9. Technical Levels as Entry Triggers
10. Risk Management at Entry
11. Sources & Further Reading
12. Gaps & Recommendations for the Options Analyzer Tool

---

## 1. Implied Volatility Rank (IVR) and IV Percentile

### Overview

Before sizing any options position, professionals answer a foundational question: is the current implied volatility (IV) elevated or depressed relative to its own recent history? The two dominant metrics for answering this are **IV Rank (IVR)** and **IV Percentile (IVP)**.

**IV Rank** is a range-normalized metric:

```
IVR = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) × 100
```

A reading of 70 means the current IV sits 70% of the way between the 52-week low and 52-week high.

**IV Percentile** is a frequency-based metric:

```
IVP = (Number of days in past year where IV < Current IV) / 252 × 100
```

A reading of 70 means IV was lower than today on 70% of trading days in the past year.

### The Critical Difference

IVR is more sensitive to outlier spikes. A single volatility event (e.g., a flash crash or earnings shock) that pushed IV to an extreme can suppress IVR readings for the entire following year even if current IV is objectively elevated. IVP is less distorted by such outliers because it measures frequency rather than range position. Platforms like Tastytrade display IVP directly on the options chain for this reason.

### Professional Thresholds

The industry has converged on a tiered framework for using these metrics:

| IVR / IVP Range | Strategy Implication |
|---|---|
| < 20 | Avoid selling premium; consider long volatility (debit spreads, calendars) |
| 20–50 | Neutral zone; reduce position size; prefer defined-risk structures |
| 50–70 | Acceptable for premium selling; standard position sizing |
| > 70 | High-conviction environment for short premium; full sizing appropriate |

Tastytrade's research, derived from analysis of over 200,000 options trades, identifies IVR > 50 as the baseline entry filter for short premium strategies including iron condors, strangles, and credit spreads. At IVR > 70, the statistical edge of premium selling is most pronounced.

Backtested data from Volatility Box shows that short iron condors entered when both IVR and IVP exceed 50 produce a 56.8% win rate across 595 tracked symbols, versus 48.2% without IV filtering. The improvement — approximately 8.6 percentage points — represents meaningful edge given the frequency of these trades.

### Academic Basis

The theoretical justification for IVR/IVP as entry filters rests on the empirical observation that implied volatility is **mean-reverting**. This was established formally by Stein (1989, *Review of Economic Studies*) and Andersen & Bollerslev (1998, *Journal of Finance*), both of whom documented that volatility is not a random walk but tends to revert toward a long-run average. When IVR is high, implied volatility is statistically more likely to fall than to rise further — directly benefiting short vega (premium-selling) positions. This is the same logic underlying GARCH models, where volatility clustering and reversion are core properties.

Natenberg's *Option Volatility and Pricing* (2nd ed., McGraw-Hill) formalizes this in practitioner terms: the correct posture is to sell options when their implied volatility exceeds the trader's estimate of future realized volatility, and to buy when implied volatility is below that estimate. IVR/IVP provides a systematic, historical-context mechanism for operationalizing this comparison.

---

## 2. IV vs. Realized Volatility (HV) — The Volatility Risk Premium

### The Volatility Risk Premium Defined

The **Volatility Risk Premium (VRP)** is the systematic tendency for implied volatility to exceed subsequently realized volatility. It represents the premium that options buyers pay — and options sellers collect — as compensation for bearing volatility risk. The VRP is arguably the most important structural feature of options markets for systematic traders.

Empirically, the VRP in S&P 500 options has averaged approximately **3–5 percentage points** of annualized volatility across studied periods. CBOE research covering 1990–2018 found that the average S&P 500 IV was 19.3% while average realized volatility was 15.1%, producing a long-run average spread of 4.2 percentage points.

### Seminal Academic Research

**Carr and Wu (2009)** — "Variance Risk Premiums," *Review of Financial Studies* — is the foundational academic treatment. Carr and Wu propose using the difference between the variance swap rate (derived from options prices) and expected future realized variance as a clean measure of the VRP. Their analysis across multiple equity indices and decades confirms the VRP is a persistent, structural feature — not a measurement artifact or regime-specific anomaly. Crucially, they also demonstrate that the VRP significantly predicts future stock returns, giving it relevance beyond pure volatility trading.

**Bollerslev, Tauchen, and Zhou (2009)** — "Expected Stock Returns and Variance Risk Premia," *Federal Reserve FEDS Working Paper* — extend this by showing the VRP predicts equity returns at quarterly horizons, establishing its macro-financial significance.

**Bakshi and Kapadia (2003)** — *Journal of Finance* — document that delta-hedged long option positions consistently lose money over time due to the VRP, providing direct evidence that the structural edge favors premium sellers.

### How Professionals Apply the VRP

A professional approach compares a real-time calculation of **IV minus HV** (historical/realized volatility) for the same underlying. Typical conventions:

- **HV period**: 20-day or 30-day close-to-close historical volatility, annualized
- **IV measure**: 30-day constant maturity IV (often the VIX for SPX, or derived from the ATM options chain for individual names)
- **Positive spread (IV > HV)**: Sell premium; the market is paying more for volatility protection than recent history suggests is warranted
- **Negative spread (IV < HV)**: Avoid short premium; consider long volatility or pass entirely

Practitioner guidance from Colin Bennett's *Trading Volatility* (a widely-circulated practitioner text) emphasizes that the IV-HV spread is directional but not sufficient alone — it should be filtered by regime (see Section 8) because the VRP can temporarily disappear or invert during risk-off environments when realized volatility spikes above implied volatility.

Quantitative approaches further refine this by forecasting realized volatility using GARCH(1,1) or HAR-RV (Heterogeneous Autoregressive Realized Volatility) models, then comparing the forecast to current IV. The spread between IV and the **forecasted** (rather than trailing) HV provides a forward-looking edge estimate. Research by Andersen, Bollerslev, Diebold, and Ebens (2001, *Journal of Finance*) establishes HAR-RV as the benchmark for realized volatility forecasting.

### Risk of the Trade

The risk for premium sellers collecting the VRP is **tail risk**. The VRP payoff distribution is negatively skewed: sellers earn small, consistent premiums most of the time but face large losses during volatility spikes. CBOE research (PUT Index study) quantifies this: an S&P 500 put-selling index generated average annual gross premiums of 37% but with drawdowns up to -800% in severe stress events. Half-Kelly or fixed-fractional sizing (see Section 5) is the professional mechanism for managing this asymmetry.

---

## 3. Term Structure of Volatility / VIX Term Structure

### What the Term Structure Reveals

The **volatility term structure** describes how implied volatility varies across expiration dates for the same underlying. When plotted, it produces a curve where each point represents the IV of a specific expiration. The shape of this curve carries significant information about the market's current regime and expectations.

**Contango** (normal): Near-term IV < longer-term IV. The curve slopes upward. This is the default state — the VIX futures curve is in contango more than 80% of the time since 2010 (per CBOE and Volatility Box analysis). Contango reflects the market's baseline expectation that current near-term uncertainty is lower than the long-run uncertainty level. Short VIX futures positions (and by extension, net short vega positions in near-term options) tend to be structurally profitable in contango environments due to positive roll yield.

**Backwardation**: Near-term IV > longer-term IV. The curve inverts. This occurs less than 20% of the time historically and typically accompanies realized stress events — earnings surprises, macro shocks, geopolitical events. In backwardation, near-term options are expensive and longer-dated options are relatively cheap. The backwardation regime is dangerous for premium sellers: it signals that the market expects near-term volatility to exceed long-run norms.

### Professional Uses of Term Structure

**Calendar spread selection**: The term structure directly determines the value of calendar spreads (long back-month, short front-month). In steep contango, front-month IV is cheap relative to back-month; buying calendars is disadvantaged. In flat or backwardated structures, front-month IV is elevated, making short-front-long-back structures attractive.

**Entry timing**: Professionals monitor the ratio of near-term to longer-term IV as a regime signal. The VIX/VIX3M ratio (30-day vs. 90-day VIX) is a commonly tracked indicator. When VIX > VIX3M (backwardation), many systematic premium sellers reduce position size or hedge more aggressively. When VIX < VIX3M (contango), the structural environment supports short-premium strategies.

**CBOE research** on the VXST/VIX ratio (9-day vs. 30-day) and the VIX/VIX6M spread documents that these ratios provide early warning of volatility regime changes.

### The Volatility Smile / Skew Connection

The term structure interacts with the **volatility skew** — the variation in IV across strikes for a given expiration. In equity markets, the skew is persistently negative: lower strikes have higher IV than higher strikes (put skew), reflecting demand for downside protection. The skew tends to steepen in backwardation (stress events) and flatten in calm contango periods.

Bates (1996, *Review of Financial Studies*) was among the first to demonstrate that neither pure stochastic volatility (Heston) nor pure jump-diffusion models alone can capture both the skew and the term structure simultaneously — a key motivation for the SVJ (Stochastic Volatility + Jumps) framework implemented in `custom_model.py`. Calibrating the Bates model to the term structure allows practitioners to price across maturities consistently and identify relative value between expirations.

---

## 4. Greeks-Based Entry Criteria

### Delta: Strike Selection and Probability of Profit

**Delta** serves dual functions in professional entry analysis: it measures directional exposure, and — for short options — it approximates the probability that the option expires in-the-money. For a short put with delta -0.30, the market implies approximately a 30% probability of the put expiring in-the-money (i.e., a 70% probability of profit at expiration).

Professional delta selection norms by strategy:

| Strategy | Typical Short Delta | Rationale |
|---|---|---|
| Short strangle legs | 0.15–0.25 | 75–85% PoP; accommodates wide range of moves |
| Credit spreads | 0.25–0.35 | Balances premium received vs. wing width |
| Cash-secured puts | 0.20–0.35 | Strike near key technical support |
| Covered calls | 0.30–0.45 | Captures more premium; accepts higher assignment risk |
| Iron condors | 0.10–0.20 per wing | High PoP; smaller credit but fewer losses |

Natenberg's *Option Volatility and Pricing* grounds this in risk-neutral probability theory: under the risk-neutral measure, delta equals N(d₂) in Black-Scholes, which is the probability of expiring ITM under that measure. In practice, due to skew and vol-of-vol, true real-world probabilities differ from risk-neutral ones — but delta remains a highly useful practical proxy.

### Theta: The Decay Curve and Optimal Entry Windows

**Theta** measures the daily time value decay of an option. The decay is non-linear: theta accelerates as expiration approaches, but the acceleration itself has a specific profile that professional traders exploit.

The mathematical relationship is that time value decays proportionally to the **square root of remaining time**. The practical implication: the last 25% of an option's life contains approximately 50% of its total time value decay. However, this final period also brings extreme **gamma acceleration** — the option's delta becomes highly sensitive to small underlying moves, creating binary-style risk.

**The 45-DTE Entry / 21-DTE Exit Framework:**

Tastytrade conducted arguably the most comprehensive empirical analysis of this question, examining over 200,000 credit spread trades across multiple market cycles. Their findings, which have become the de-facto professional standard for systematic premium selling:

- **Entry at 45 DTE (days to expiration)**: Captures the steepening portion of the theta decay curve while avoiding extreme gamma risk
- **Exit at 21 DTE or 50% of max profit**: Exits before the final gamma-acceleration phase; improves risk-adjusted returns by approximately 15–20% compared to holding to expiration
- **The 21-DTE gamma threshold**: After 21 DTE, gamma risk per dollar of remaining theta increases sharply enough that the risk/reward of holding deteriorates

This is consistent with the theoretical work of Carr and Madan on the time value decay profile of European options, and with practitioner analysis by Karen Szala (former CME executive) and subsequent Tastytrade research published between 2012–2020.

### Vega: Sizing Volatility Exposure

**Vega** measures the dollar change in an option's value for a 1-percentage-point change in implied volatility. Professionals use vega to:

1. **Size positions relative to portfolio**: Total portfolio vega is capped as a percentage of notional value. A common institutional guideline is keeping net vega exposure below 0.5–1.0% of portfolio value per 1-vol-point move.
2. **Match vega to IV environment**: High IVR environments call for short vega (selling options); low IVR environments for long vega (buying options or calendars).
3. **Balance vega across expirations**: Spreading vega across multiple expiration cycles reduces event-specific volatility risk.

Colin Bennett's *Trading Volatility* provides the most thorough practitioner treatment of vega budgeting, describing how market-making desks assign vega limits per underlying and per portfolio sector.

---

## 5. Expected Value / Edge-Based Entry

### The Professional EV Framework

Professional options traders do not enter positions based on directional conviction alone — they evaluate whether the **expected value (EV)** of the trade is positive given the probabilities implied by the options market and their own assessments. The EV of a short option position is:

```
EV = (PoP × Premium Collected) - ((1 - PoP) × Expected Loss)
```

Where `PoP` is derived from delta (or from Monte Carlo simulation), and `Expected Loss` accounts for the magnitude of losses in adverse scenarios — not just the probability of loss.

This distinction matters enormously: a position with 85% PoP can have negative EV if the 15% loss scenarios are catastrophically large. This is why EV analysis requires full probability distribution analysis rather than just PoP at expiration.

### Kelly Criterion and Fractional Kelly

The **Kelly Criterion** (Kelly, 1956, *Bell System Technical Journal*) provides a theoretical framework for optimal position sizing given a known edge:

```
f* = (bp - q) / b
```

Where:
- `f*` = fraction of capital to allocate
- `b` = net odds (reward-to-risk ratio)
- `p` = probability of winning
- `q` = 1 - p = probability of losing

**Direct Kelly application to options is problematic** for several reasons:
1. Options payoffs are not binary — the loss distribution has a fat left tail
2. Kelly assumes known, stable probabilities — options markets have estimation error
3. Full Kelly sizing produces extreme drawdowns due to estimation errors (Thorp, 1969, *Review of the International Statistical Institute*)

**Professional adaptations:**

- **Half-Kelly (50% of f*)**: The most common professional convention. Half-Kelly captures approximately 75% of the theoretical growth rate while dramatically reducing drawdown risk. The RiverPark Structural Alpha white paper (Berman, 2014) uses half-Kelly as the sizing baseline for systematic volatility-selling strategies.
- **Quarter-Kelly (25% of f*)**: Used by more conservative systematic funds; prioritizes capital preservation over growth rate
- **Fixed fractional (1–2% per trade)**: A simpler alternative that avoids Kelly's estimation requirements. Most retail-facing professional frameworks (Tastytrade, Option Alpha) recommend this: no single position should risk more than 1–5% of total portfolio capital

Alpha Theory's analysis of Kelly in practice identifies that the primary source of Kelly overfitting is **overestimating edge** — the win rate (p) and expected return (b) used in the formula are typically higher than realized values due to backtest overfitting or regime change.

### Probability-Weighted EV Calculations

More sophisticated practitioners use **scenario trees or Monte Carlo-derived probability distributions** to calculate EV across a range of outcomes rather than a binary win/loss. For example:

- P(underlying within short strikes at expiration) × full credit kept
- P(underlying outside short strikes but within long strikes) × partial loss based on interpolated payoff
- P(underlying outside long strikes) × max defined loss

This approach — matching the payoff function precisely — is what professional risk systems at market-making firms and hedge funds use. It is also what `custom_model.py` approximates with its 100,000-path Monte Carlo simulation.

---

## 6. Liquidity Filters

### Why Liquidity Matters Before Entry

Entering an illiquid options position has two direct costs: (1) paying a wide bid-ask spread on entry, and (2) paying a wide spread again on exit, potentially at a worse time. A position with a 20% bid-ask spread relative to premium requires the underlying thesis to be correct by more than 20% just to break even on the round trip. Liquidity filters are therefore not cosmetic — they directly determine the net expected value of the trade.

### Professional Thresholds

**Bid-Ask Spread Standards:**

The industry-standard filter is that the bid-ask spread should not exceed **10% of the option's mid-price** for liquid names, with a hard upper limit of **$0.30** width for options priced above $3.00. More precisely:

| Option Price Range | Maximum Acceptable Spread |
|---|---|
| < $1.00 | $0.10 |
| $1.00–$3.00 | $0.20 |
| $3.00–$10.00 | $0.30 |
| > $10.00 | < 5% of mid-price |

For spread strategies (iron condors, credit spreads), the aggregate spread across all four legs compounds. A four-legged strategy with $0.20 average per-leg spread costs $0.80 in transaction friction — significant relative to a $1.50 credit.

**Open Interest Thresholds:**

- **Minimum**: 100 OI; below this, spreads are typically prohibitive (> $0.20 per leg)
- **Practical minimum for systematic traders**: 500 OI; $0.03–$0.10 spreads become achievable
- **High-liquidity target**: 1,000+ OI at the specific strike; penny-wide or near-penny spreads
- **Underlying-level filter**: Underlyings with fewer than 10,000 total OI across all strikes/expirations are generally avoided

**Volume Requirements:**

Daily volume above 100 contracts at a specific strike/expiration suggests active price discovery. Volume-to-open-interest ratios above 0.10 (10%) are considered healthy. CBOE market structure research notes that off-screen liquidity (institutional block trades) supplements visible volume in major names like SPY, QQQ, and large-cap single names — meaning volume understates true depth for those instruments.

**Underlying Liquidity:**

Professionals filter the universe of tradable underlyings before individual strike analysis:
- Average daily equity volume > 500,000 shares (ensures options market maker hedging is viable)
- Listed on major exchange with multiple market makers competing
- Preferably included in major index (S&P 500, Nasdaq 100) for institutional sponsorship

### Slippage Modeling

Sophisticated tools model expected slippage as a function of position size relative to average daily volume. For options, a standard rule is that order size should not exceed 1–2% of average daily volume without expecting meaningful adverse market impact.

---

## 7. Earnings / Event-Driven Analysis

### The Implied Move Framework

Earnings announcements are the most common **binary event** in options markets. The standard professional approach to earnings involves:

1. **Calculating the implied move** from options prices
2. **Comparing to historical actual moves**
3. **Deciding whether to own or sell the implied move**

**Calculating the Implied Move:**

Two standard methodologies exist:

*Method 1 — ATM Straddle (Simple):*
```
Implied Move = Front-Expiry ATM Straddle Price × 0.85
```
The 0.85 multiplier accounts for the fact that the straddle slightly overstates the expected move due to how options are priced across the full distribution. Example: AAPL at $230, front-week ATM straddle at $14.00 → implied move = $14.00 × 0.85 = $11.90 (±5.2%).

*Method 2 — Weighted Average (Tastytrade / Professional):*
```
EM = (0.60 × ATM Straddle) + (0.30 × 1st OTM Strangle) + (0.10 × 2nd OTM Strangle)
```
This weighted average incorporates information from the options skew to produce a more accurate range estimate.

### Implied vs. Actual Move Analysis

The key analytical question is: **is the implied move historically overpriced or underpriced?**

Research from ORATS (Options Research and Technology Services) and practitioner studies consistently show:
- Stocks stay within their earnings expected move approximately **70–75% of the time** (vs. the theoretical 68% for a 1-standard-deviation move)
- This slight overperformance reflects that IV is systematically inflated heading into earnings — the market charges extra for binary event risk beyond what a normal distribution would suggest
- This overpricing is the **earnings volatility risk premium** — a specific manifestation of the VRP discussed in Section 2

**How to exploit this:**

*When implied move > 1.5× historical average actual move*: Consider selling the earnings move via short straddle or strangle (typically closed same day or next day). The premium received exceeds what the typical move warrants.

*When implied move < 0.8× historical average actual move*: Consider buying the earnings move via long straddle or wide strangle. Options are historically cheap given the typical post-earnings move.

### IV Crush and Its Timing

A critical phenomenon for all event-driven options traders is **IV crush**: the collapse of implied volatility immediately after an earnings announcement (or other binary event) as uncertainty resolves. Professional traders who hold short straddles or strangles through earnings benefit from IV crush — even if the underlying moves more than expected, the IV collapse partially offsets the intrinsic value increase.

Conversely, traders who buy options ahead of earnings must overcome both the premium paid AND the IV crush to profit. The standard professional response is to never buy near-term options more than 1–2 days before earnings without a directional conviction specifically about the magnitude of the move.

**Pre-earnings IV inflation** follows a documented pattern: IV begins rising approximately 10–14 days before earnings, accelerating in the final 3–5 days, then collapsing within minutes of the announcement. ORATS tracks this pattern systematically and provides earnings IV history for individual names.

### The IV-Crush Magnitude Estimate

Professionals estimate post-event IV by comparing the current front-expiry IV to the second-expiry IV, treating the difference as the "event premium." After the event, front-expiry IV converges toward the second-expiry level. This is used to estimate how much IV crush will occur even if the underlying moves unexpectedly.

---

## 8. Sector / Macro Context

### Why Macro Regime Matters for Options

Options strategies — particularly short-volatility strategies — are highly sensitive to macro regime. In **risk-on** environments, realized volatility tends to be low, the VRP is large, and premium selling is consistently profitable. In **risk-off** environments, realized volatility spikes, the VRP compresses or inverts, and short premium strategies suffer large drawdowns. Ignoring macro regime is one of the most common errors among retail options traders and is systematically addressed by institutional practitioners.

### Regime Classification Framework

Professional systems classify the macro environment along several dimensions:

**Volatility Regime** (based on VIX level and trend):
- Low vol (VIX < 15): Tight spreads, small premiums; reduce position size for short-premium strategies
- Normal vol (VIX 15–25): Standard environment; full-size positions appropriate
- Elevated vol (VIX 25–35): High premiums but elevated risk; half-size or defined-risk only
- Crisis vol (VIX > 35): Avoid net short vega; consider long volatility or protective positions

**Risk-On / Risk-Off Signals:**
- Credit spreads (high-yield vs. investment-grade): Widening signals risk-off
- Equity/bond correlation: Positive correlation (both falling) = stress signal
- VIX/VIX3M term structure: Inversion (backwardation) = risk-off warning
- USD strength vs. EM currencies: USD strengthening broadly = risk-off

### Sector Rotation and Options Strategy Alignment

Sector rotation — the cyclical movement of institutional capital between equity sectors — creates relative value opportunities in single-stock options. Professional sector-rotation analysis identifies:

**Early cycle (recovery)**: Financials, consumer discretionary, industrials lead. Elevated individual stock volatility as earnings estimates reset higher → favor premium selling on individual names in these sectors.

**Mid cycle (expansion)**: Technology, materials, energy lead. Realized vol low, trend strong → bull call spreads, covered calls; reduce strangle selling.

**Late cycle / contraction**: Utilities, healthcare, consumer staples outperform. Cross-sector correlation rises, reducing diversification benefit of multi-underlying options books → reduce single-name size, shift to index options.

CME Group research on equity market rotation documents that sector-level options IV diverges significantly during rotation periods, creating IV discrepancies across sectors that inform relative value strategies.

### Correlation and Dispersion Trading

When cross-asset and cross-sector correlation rises (typically risk-off), index implied volatility rises faster than the average of individual stock implied volatilities. This creates a **dispersion trade opportunity**: sell index vol, buy single-stock vol. The professional measure is:

```
Realized Dispersion = Avg(σ_individual) - σ_index
```

Positive dispersion favors long single-stock straddles vs. short index straddles. This strategy is primarily institutional but the underlying monitoring (tracking correlation levels) is relevant for all professional options practitioners when sizing portfolio exposure.

---

## 9. Technical Levels as Entry Triggers

### The Integration of Technicals and Options Analysis

Technical analysis — support/resistance levels, moving averages, trend channels, and volume nodes — serves a specific and practical role in options entry: **helping identify the strike prices and expiration windows most consistent with the underlying's likely price path.**

For premium sellers, the ideal short strike is one that coincides with a strong technical level — a strike beyond which the underlying is unlikely to move. For directional buyers, the ideal long strike is one where the underlying has technical catalyst support.

### Support/Resistance as Strike Anchor

The most common integration is using technical support as the anchor for short put strikes:

1. Identify technical support level from chart analysis (prior highs, moving averages, Fibonacci retracements, volume-by-price nodes)
2. Select the options strike **at or below** the support level
3. Verify the strike is at least 1 standard deviation below current price (consistent with < 0.25 delta)
4. Sell the put or put spread at that strike

Example: If XYZ is trading at $100 and has clear chart support at $85, a professional might sell the $85 put (roughly 1.5 standard deviations OTM) — they have both statistical probability (delta-based) AND fundamental technical justification for the level.

### Round-Number Convergence

A well-documented phenomenon in options markets is the concentration of open interest at round-number strikes — $50, $100, $150, $200, etc. This concentration exists precisely because market participants and technical analysts focus on these levels. Research by Biais and Hillion (1994, *Review of Financial Studies*) and subsequent work documents that options strikes at round psychological levels have systematically higher open interest and often act as **price magnets** or barriers.

Professionals exploit this by:
- Selling strikes at major round numbers where open interest concentration suggests market makers have large hedging interest (these levels tend to be "sticky")
- Avoiding strikes in thin air between technical levels where price can move without friction

### Moving Averages and Expiration Selection

The **200-day moving average** is the most widely monitored long-term technical level in institutional markets. When selecting expiration, professionals often target expirations that expire before or around anticipated MA tests — allowing theta to decay if the MA holds, or defining the loss clearly if it breaks.

The 20-day and 50-day moving averages serve the same function for shorter-duration strategies (30–45 DTE).

### Volume Profile and High-Volume Nodes

Market Profile and Volume Profile analysis identifies price levels with historically high trading volume — "high-volume nodes" — which represent price levels where broad consensus exists between buyers and sellers. These nodes act as strong support/resistance and are particularly valuable for options strike selection because: (1) they represent actual transacted prices rather than derived indicator levels, and (2) institutional memory around these levels creates recurring support/resistance.

---

## 10. Risk Management at Entry

### The Three Dimensions of Entry-Level Risk Management

Professional risk management at entry has three dimensions: **individual position sizing**, **portfolio-level Greeks management**, and **correlation / regime adjustment**.

### Individual Position Sizing

**Fixed-fractional sizing** is the most common professional framework:

- No single options position should risk more than **1–5% of total portfolio capital** on a max-loss basis
- Typical institutional target: 1–2% per trade (retail practitioners often allow 2–5%)
- "Max loss" for defined-risk positions: width of the spread minus premium collected × contracts
- "Max loss" for undefined-risk positions (strangles, naked puts): estimated at 3–5× premium collected, based on historical tail loss analysis

**Example calculation:**
- Portfolio: $100,000
- Max per-trade risk: 2% = $2,000
- Iron condor with $5 wide wings, $1.50 credit: max loss = ($5.00 - $1.50) × 100 = $350 per contract
- Max contracts = $2,000 / $350 = 5.7 → enter 5 contracts

This sizing prevents any single event from causing portfolio-threatening damage.

### Portfolio-Level Greeks Management

Institutional practitioners manage risk at the **portfolio Greeks** level, not just the individual position level. The key portfolio-level metrics:

**Beta-Weighted Portfolio Delta:**
Converting each position's delta to SPY-equivalent exposure allows an apples-to-apples comparison of directional risk across all holdings. Most full-time options traders managing diversified books target **near-zero beta-weighted delta** — not because they have no directional view, but because staying delta-neutral removes the largest source of risk and allows theta decay and volatility premium to drive P&L.

```
Beta-Weighted Delta of position = Position Delta × Stock Beta × (Stock Price / SPY Price)
```

CBOE's guide on beta-weighting (2022) documents that institutional hedges are sized using beta-weighted delta to ensure the hedge correctly offsets portfolio-level directional exposure.

**Portfolio Vega:**
Net portfolio vega measures total sensitivity to a uniform 1-vol-point shift in IV. Professional targets:
- Keep net portfolio vega between -0.5% and -1.5% of portfolio value per 1-vol-point (short vega bias is normal for premium-focused books)
- Avoid net long vega in low-IVR environments (buying expensive optionality)
- Avoid net short vega positions that exceed twice normal size during elevated VIX (backwardated term structure)

**Portfolio Theta:**
Target daily theta income of 0.1–0.3% of portfolio value for systematic income-generation strategies. Higher daily theta implies proportionally higher gamma risk.

**Gamma Exposure:**
Portfolio gamma becomes the critical metric as positions approach expiration. Professionals limit overall net negative gamma below $X per 1% underlying move based on portfolio size. Approaching expiration, gamma management often dominates all other considerations.

### Correlation Risk and Position Concentration

A common professional error is constructing a portfolio of high-IV stocks with good individual-position metrics, only to find that the positions are all correlated — they all respond similarly to the same macro shock.

**Correlation management practices:**

1. **Limit sector concentration**: No more than 20–25% of portfolio risk in a single GICS sector
2. **Underlying correlation screens**: Avoid holding short-vol positions simultaneously on high-correlation pairs (e.g., XOM and CVX, or AAPL and MSFT at similar position sizes)
3. **Portfolio stress testing**: Model the portfolio P&L under scenarios of SPX -10%, VIX +10 points, sector rotation events
4. **Maximum number of positions**: Tastytrade recommends 10–15 simultaneous uncorrelated positions as the practical limit for a single manager maintaining active oversight

### Max Loss Definition Before Entry

Every professional trade requires a **pre-defined max loss scenario** and a clear exit rule before the position is opened:

- **Defined-risk positions** (vertical spreads, iron condors): Max loss is the width of the spread minus premium received — automatically known at entry
- **Undefined-risk positions** (short strangles): Set a hard delta or loss trigger (e.g., "close if loss reaches 2× premium collected" or "close if short strike delta reaches 0.50")
- **Stop-loss levels**: Typically 100–200% of premium received; at that point, the statistical edge has eroded and the position becomes a speculation on recovery rather than a premium-collection trade

The practice of defining max loss before entry — rather than managing emotionally during adverse moves — is one of the most consistent differences between professional and retail options traders documented in behavioral finance research (Odean, 1998, *Journal of Finance* on the disposition effect in retail trading).

---

## Sources & Further Reading

### Books (Foundational Practitioner Literature)

- Natenberg, Sheldon. *Option Volatility and Pricing: Advanced Trading Strategies and Techniques*, 2nd ed. McGraw-Hill, 1994. — The standard starting text at most options market-making firms worldwide. Covers theoretical pricing, volatility, and Greeks management from first principles.

- Bennett, Colin. *Trading Volatility: Trading Volatility, Correlation, Term Structure and Skew*. Self-published / Santander trading desk, 2012. — The most comprehensive practitioner text on volatility trading, including VRP, term structure, skew, and correlation strategies.

- Cottle, Charles, and Peter Lusk. *Options: Perception and Deception*. — Advanced professional-level treatment of options positions, risk, and adjustment.

- Passarelli, Dan. *Trading Option Greeks: How Time, Volatility, and Other Pricing Factors Drive Profit*. Bloomberg Press, 2012. — Detailed treatment of Greeks-based position management.

### Academic Papers (Peer-Reviewed)

- **Carr, P., and L. Wu (2009).** "Variance Risk Premiums." *Review of Financial Studies*, 22(3), 1311–1341. — Definitive academic treatment of the VRP across multiple markets.

- **Bates, D.S. (1996).** "Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options." *Review of Financial Studies*, 9(1), 69–107. — Foundation of the SVJ model (Bates model) used in the tool.

- **Heston, S. (1993).** "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327–343. — Foundation of the Heston model used in the tool.

- **Black, F., and M. Scholes (1973).** "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654. — The foundational options pricing model.

- **Bollerslev, T., G. Tauchen, and H. Zhou (2009).** "Expected Stock Returns and Variance Risk Premia." *Federal Reserve FEDS Working Paper 2007-11*. — Documents VRP's predictive power for equity returns.

- **Andersen, T.G., T. Bollerslev, F.X. Diebold, and H. Ebens (2001).** "The Distribution of Realized Stock Return Volatility." *Journal of Financial Economics*, 61(1), 43–76. — Establishes realized volatility measurement and the HAR-RV forecasting model.

- **Bakshi, G., and N. Kapadia (2003).** "Delta-Hedged Gains and the Negative Market Volatility Risk Premium." *Review of Financial Studies*, 16(2), 527–566. — Documents that the VRP causes delta-hedged long option positions to lose systematically.

- **Kelly, J.L. (1956).** "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4), 917–926. — Original Kelly Criterion paper.

- **Odean, T. (1998).** "Are Investors Reluctant to Realize Their Losses?" *Journal of Finance*, 53(5), 1775–1798. — Behavioral finance evidence on disposition effect in retail trading, relevant to stop-loss adherence.

- **Jones, C.S., and J. Wang (2012).** "The Term Structure of Equity Option Implied Volatility." *Working Paper, USC*. — Documents how the IV term structure behaves across market regimes.

### Practitioner Research and Institutional Sources

- **CBOE Research** — "New Research Shows Options-Based Strategies Can Generate Higher Gross Premiums" (2019). Documents the PUT Index VRP harvesting performance 1986–2018.

- **CBOE Insights** — "Inside Volatility Trading: Is VIX Backwardation a Sign of a Future Down Market?" — Analysis of VIX term structure regimes and their predictive value.

- **Tastytrade Research** (tasty.works / tastylive) — Empirical studies on 45-DTE entry / 21-DTE exit, IVP thresholds for iron condors, and portfolio-level management. Based on 200,000+ live trades. Primary practitioner reference for systematic retail-level options trading.

- **ORATS (Options Research and Technology Services)** — Earnings IV history, term structure analysis, and IV vs. historical vol databases. Commercial provider used by professional options traders for systematic research.

- **Volatility Box Research** — "IV Rank vs IV Percentile: The Definitive Comparison" and "VIX Futures: Contango, Backwardation, and Roll Yield." Backtested performance of IV-filtered strategies across 595 symbols.

- **RiverPark Structural Alpha White Paper** (Berman, 2014) — "The Benefits of Systematically Selling Volatility." Institutional white paper documenting the VRP-harvesting strategy with Kelly-based sizing.

- **Quantpedia** — "Volatility Risk Premium Effect" and "Exploiting Term Structure of VIX Futures." Quantified strategy summaries with academic citations and backtest statistics.

---

## Gaps & Recommendations for the Options Analyzer Tool

### What the Current Tool Does Well

`custom_model.py` already implements several professional-grade capabilities that distinguish it from basic options calculators:

- **Multi-model pricing**: Black-Scholes, Heston (stochastic volatility), and Bates (SVJ) cover the full range of industry-standard pricing frameworks, from the simplest to the academically most complete
- **Monte Carlo simulation (100,000 paths)**: Enables full distributional analysis of P&L, not just point estimates — directly supporting EV calculation (Section 5)
- **Complete Greeks**: Delta, Gamma, Theta, Vega, Rho are calculated; these are the inputs for all position-level risk management described in Section 4 and 10
- **Implied vol extraction**: Solves for market-implied volatility from observed prices — the starting point for VRP analysis (Section 2)
- **Realized vol calculation**: Enables the IV vs. HV spread calculation that quantifies the VRP edge
- **IV vs. HV comparison**: Explicitly implemented — provides the core edge indicator from Section 2
- **Probability of Profit and Expected Value**: Outputs aligned with the professional EV framework of Section 5
- **Reward:Risk ratio**: Standard position evaluation metric
- **Auto-profile selection based on market cap/sector**: Partial macro-context implementation (Section 8)
- **3-tier exit framework**: Addresses Section 10 (risk management at exit)

### Priority Gaps — Ranked by Professional Impact

The following features are identified as gaps, ranked from highest to lowest impact on professional-quality entry analysis:

---

**GAP 1 — No IVR / IV Percentile Calculation [CRITICAL]**

*What professionals use it for:* Primary filter before any premium-selling entry (Section 1). Without this, the tool cannot determine whether current IV is elevated or depressed relative to history.

*What's needed:*
- Download 52-week daily IV history for the target underlying (available via yahooquery or a data provider)
- Calculate both IVR (range-normalized) and IVP (frequency-based) from this history
- Display with threshold annotations (< 30 = low, 30–50 = neutral, 50–70 = favorable for selling, > 70 = high-conviction sell)

*Suggested implementation:* Add a `get_iv_history()` function using yahooquery or yfinance to pull 252 days of ATM IV, then compute IVR and IVP. Cost: ~30 lines of code plus API calls.

---

**GAP 2 — No Liquidity Filter (Bid-Ask Spread Quality + OI/Volume) [HIGH]**

*What professionals use it for:* Before entering any position, verify the market is liquid enough that the round-trip friction does not destroy the edge (Section 6).

*What's needed:*
- Fetch bid-ask spread for the specific option being analyzed
- Calculate spread as % of mid-price
- Flag if spread exceeds 10% of mid-price or $0.30 absolute
- Fetch OI and daily volume for the specific strike/expiration
- Warn if OI < 500 or volume < 100

*Suggested implementation:* The Yahoo Finance options API (already used in the tool) returns bid, ask, volume, and openInterest for each contract — these fields just need to be surfaced and tested against thresholds in the output.

---

**GAP 3 — No Earnings Date Awareness [HIGH]**

*What professionals use it for:* Earnings events fundamentally change the risk/reward of options positions. Not knowing an earnings date can lead to unintended exposure through a binary event (Section 7).

*What's needed:*
- Look up the next earnings date for the underlying (yahooquery exposes `calendar_events`)
- If an earnings date falls within the selected expiration window, display a warning
- Calculate the implied move using the ATM straddle method
- Compare implied move to historical average earnings moves (if historical earnings history is available)

*Suggested implementation:* yahooquery's `Ticker.calendar_events` returns next earnings date. ATM straddle price is already being calculated by the tool during IV extraction — the implied move calculation adds one line.

---

**GAP 4 — No Term Structure Analysis [HIGH]**

*What professionals use it for:* Determines the volatility regime (contango/backwardation), informs expiration selection, and identifies calendar spread opportunities (Section 3).

*What's needed:*
- Fetch IVs for at least 3 expirations (front-month, ~2-month, ~3-month)
- Calculate the term structure slope (front / back IV ratio)
- Flag backwardation (ratio > 1.0) as a risk-off signal
- Display term structure visualization or tabular comparison

*Suggested implementation:* The Yahoo Finance options API returns data for multiple expirations. Extracting ATM IV for each and computing ratios adds modest complexity.

---

**GAP 5 — No Implied Move vs. Historical Earnings Move Comparison [MEDIUM-HIGH]**

*What professionals use it for:* The core edge identification tool for event-driven trades (Section 7). Tells whether the market is over- or under-pricing the expected move.

*What's needed:*
- Calculate implied move: `ATM_straddle_price × 0.85` (or weighted average method)
- Source historical earnings moves for the underlying (several free sources via web scraping or paid APIs like ORATS, Market Chameleon)
- Display: current implied move vs. average historical move vs. max historical move

*Limitation:* Historical earnings move data requires an external source (Market Chameleon has free tiers; ORATS is professional-grade but paid).

---

**GAP 6 — No Portfolio-Level Greeks Tracking [MEDIUM]**

*What professionals use it for:* Ensures no single entry creates an unacceptable portfolio-level risk concentration (Section 10). Prevents the scenario of 10 apparently uncorrelated positions all being net short vol.

*What's needed:*
- A session-level portfolio state: list of open positions with their Greeks
- Sum portfolio delta, gamma, theta, vega across all positions
- Beta-weight portfolio delta to SPX equivalent
- Flag if portfolio vega exceeds -1.5% of portfolio capital per vol point

*Suggested implementation:* Maintain a JSON file or in-session dictionary of open positions, update on each new analysis, display aggregate Greeks in a portfolio summary section.

---

**GAP 7 — No Sector / Macro Regime Integration [MEDIUM]**

*What professionals use it for:* Adjusts position sizing and strategy selection based on the macro environment (Section 8). Prevents full-size short-vol positions during risk-off regimes.

*What's needed:*
- Fetch VIX level and VIX term structure (VIX vs. VIX3M)
- Classify regime: Low Vol / Normal / Elevated / Crisis
- Fetch sector ETF performance (e.g., XLK, XLF, XLE) to identify rotation
- Apply sizing adjustment: 100% size in Normal, 75% in Elevated, 50% or skip in Crisis

*Suggested implementation:* VIX is fetchable via yahooquery as ticker "^VIX"; VIX3M as "^VIX3M". A simple regime table can hardcode the thresholds.

---

**GAP 8 — No IVR-Adjusted Position Sizing [MEDIUM]**

*What professionals use it for:* Scales position size proportionally to edge signal strength — larger size in high-IVR environments, smaller in low-IVR (Sections 1 and 5).

*What's needed:*
- Once IVR is calculated (Gap 1), apply a sizing scalar:
  - IVR < 30: 50% of base size
  - IVR 30–50: 75% of base size
  - IVR 50–70: 100% of base size
  - IVR > 70: 125% of base size (or full size, capped by Kelly/fixed-fractional)

*Suggested implementation:* A single multiplier function applied to the contracts output, once IVR is calculated.

---

**Summary Table**

| Gap | Section | Priority | Estimated Implementation Complexity |
|---|---|---|---|
| IVR / IV Percentile | 1 | Critical | Medium (requires IV history data) |
| Liquidity filter (bid-ask, OI, volume) | 6 | High | Low (data already fetched) |
| Earnings date awareness | 7 | High | Low (yahooquery calendar_events) |
| Term structure analysis | 3 | High | Medium (multi-expiry IV fetch) |
| Implied move vs. historical earnings | 7 | Medium-High | High (requires external earnings history) |
| Portfolio-level Greeks tracking | 10 | Medium | Medium (requires session state) |
| Sector / macro regime integration | 8 | Medium | Medium (VIX data, sector ETFs) |
| IVR-adjusted position sizing | 1, 5 | Medium | Low (depends on Gap 1) |

---

*Document prepared for `custom_model.py` feature development. Research synthesis covers academic literature through 2024 and practitioner standards current as of 2026. All IVR thresholds, DTE guidelines, and Greeks targets are empirical conventions, not mathematical certainties — they reflect consensus professional practice derived from large-scale backtesting and institutional research.*
