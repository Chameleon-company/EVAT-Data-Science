# EVAT — Forecasting Beyond the Dataset (3‑Hour Bins)

**Date:** 7 Sep 2025 (AEST)  
**Context:** Your current notebook (`EVAT_Congestion_GoldenPath_3h_with_baselines-updated.ipynb`) builds 3‑hour arrival counts per station, evaluates baselines (Naïve / Seasonal‑Naïve / Moving Average), fits classical models (Prophet, SARIMA) and an LSTM for count data, and converts predicted arrivals into queueing metrics (M/M/c via Erlang‑C).  
**Goal:** Produce **future forecasts** (dates *beyond* the last timestamp in your dataset) and propagate them through the queueing layer for scenario testing.

---

## 1) What your notebook already does (condensed)
- **Aggregation**: Session → 3‑hour bins (BIN_MIN = 180) per station to reduce sparsity; 8 bins/day.
- **Feature Engineering**: Calendar (hour‑of‑day, day‑of‑week), perhaps lags/rolling stats.
- **Split**: Time‑based (e.g., 70/15/15) to avoid leakage.
- **Baselines**: 
  - **Naïve** (y_t = y_{t-1}), 
  - **Seasonal‑Naïve** (y_t = y_{t-8} for daily seasonality),
  - **Moving Avg** (k ~ 4 bins = half‑day).
- **Classical models**: Prophet & SARIMA per station, evaluated on val/test.
- **Neural model**: LSTM on sequences of length `SEQ` (e.g., 16 bins ≈ 2 days) targeting 3‑hour counts, with non‑negativity via activation (e.g., `softplus`) and/or Poisson‑style loss.
- **Queueing**: Convert predicted 3‑hour counts → hourly lambda; robust per‑station mu; data‑driven **c**; compute utilization rho and expected wait E[Wq] (Erlang‑C).

**What’s missing for *future* dates:** A clean, reusable pipeline that **builds a future time index**, **generates multi‑step forecasts** from each model, **merges results** per station, and **runs queueing** on those future bins.

---

## 2) Big‑picture approach (multi‑step forecasting)
You need **H future bins** per station (e.g., 7 days → H = 56 bins since 8 bins/day). We’ll support **four model routes** and an optional **ensemble**:

1. **Baselines (multi‑step)** — trivial to extend:
   - Naïve: yhat_{t+h} = y_t (constant forecast).
   - Seasonal‑Naïve (daily): yhat_{t+h} = y_{t+h-8}.
   - Moving Avg: recursively apply rolling mean from last k values.

2. **Prophet (per station)** — make a future dataframe with 3‑hour frequency and predict H bins ahead.

3. **SARIMA (per station)** — fit with daily seasonality s=8 (optional weekly s=56) and use `.get_forecast(steps=H)`.

4. **Poisson‑LSTM (recursive)** — start from the last `SEQ` bins, predict next bin, append it, repeat H times. This is the simplest upgrade that reuses your trained single‑step model. (A more advanced alternative is a **multi‑output** LSTM trained to emit H steps in one shot.)

5. **Ensemble (optional)** — blend forecasts per station (e.g., weights proportional to recent validation performance or use a meta‑learner).

Finally, **map counts → queue inputs** for each future bin: lambda_hour = yhat_3h / 3, combine with mu and c for **Erlang‑C**.

---

## 3) Concrete steps

### Step A — Define your *future* time index
- Choose **horizon** `H` in bins (e.g., 14 days → `H = 14*8 = 112`).
- For each station, build a `pd.date_range(start=last_ts + 3h, periods=H, freq='3H')`.
- Create a future frame with calendar features (hour‑of‑day, day‑of‑week, holiday flags used by models).

### Step B — Forecasting by model

**B1. Baselines (vectorized):**
- Take the last `k` bins from the station’s history and produce H recursive predictions:
  - Naïve: repeat the last value.
  - Seasonal‑Naïve: copy values from 1 day (8 bins) or 1 week (56 bins) back.
  - Moving Avg: rolling mean over trailing `k` (recomputed as you append forecasts).

**B2. Prophet:**
- Fit per station on `ds = timestamp`, `y = arrivals`.
- `future = model.make_future_dataframe(periods=H, freq='3H')`.
- `forecast = model.predict(future)` → take the last H rows for out‑of‑sample predictions.

**B3. SARIMA:**
- Fit per station with seasonal period **s=8** (daily). If weekly effects are strong, consider s=56 or add exogenous regressors.
- `res.get_forecast(steps=H)` → use `.predicted_mean` (and optionally confidence intervals).

**B4. Poisson‑LSTM (recursive autoregression):**
- Keep the scaler/normalizer consistent with training.
- Start with the last `SEQ` bins of the *scaled* series; predict next yhat_{t+1}; invert scaling; append to the history; rescale; repeat H times.
- If your LSTM outputs a **rate** (lambda_hat) under Poisson loss, that is the expected count for the 3‑hour bin. Clip to `>=0` and keep as float; integer rounding can be done only at display time.

> **Tip:** For stability, you can dampen drift by mixing each recursive step with a seasonal prior, e.g., `yhat = 0.8 * yhat_LSTM + 0.2 * y_seasonal_naive`.

### Step C — Optional ensemble per station
- Compute validation MAE/RMSE per model on the latest window.
- Convert to weights (e.g., inverse RMSE normalized) and blend per‑bin forecasts.

### Step D — Queueing for future bins
- **lambda_hour** = yhat_3h / 3
- **mu**: use your existing robust estimator (e.g., per‑station median service rate by time‑of‑day).
- **c**: fixed from assets metadata or data‑driven peak concurrency; allow **what‑if** multipliers.
- Compute **Erlang‑C** wait E[Wq] and **rho = lambda / (c·mu)** for each future bin.

---

## 4) Minimal code snippets (drop‑in helpers)

> These are **templates**; align names/scalers with your notebook.

### A. Build a future index (per station)
```python
def make_future_bins(last_ts, H, freq='3H'):
    # Build H future timestamps after the last observed bin
    return pd.date_range(start=last_ts + pd.to_timedelta(freq), periods=H, freq=freq)
```

### B. Recursive LSTM multi‑step forecast
```python
def lstm_recursive_forecast(model, last_seq_scaled, scaler, H):
    """
    model: trained Keras model that predicts next scaled value from last SEQ steps
    last_seq_scaled: np.array shape (SEQ,) of the *scaled* last values
    scaler: fitted scaler with .inverse_transform and .transform on shape (n,1)
    H: number of future bins
    """
    seq = last_seq_scaled.reshape(1, -1, 1)  # (1, SEQ, 1)
    preds = []
    for _ in range(H):
        yhat_scaled = model.predict(seq, verbose=0)          # (1,1)
        yhat = scaler.inverse_transform(yhat_scaled)[0, 0]   # back to counts
        yhat = max(0.0, float(yhat))                         # non‑negative
        preds.append(yhat)
        # roll window: append scaled yhat and drop oldest
        next_scaled = scaler.transform([[yhat]])[0, 0]
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = next_scaled
    return np.array(preds)
```

### C. Prophet forecast (per station)
```python
from prophet import Prophet

def prophet_forecast(series_df, H):
    # series_df: columns ['timestamp','arrivals'] for one station
    df = series_df.rename(columns={'timestamp':'ds', 'arrivals':'y'})
    m = Prophet()  # add seasonality/holidays if you used them in training
    m.fit(df)
    future = m.make_future_dataframe(periods=H, freq='3H')
    fc = m.predict(future)
    out = fc.tail(H)[['ds','yhat']].rename(columns={'ds':'timestamp', 'yhat':'yhat_prophet'})
    return out
```

### D. SARIMA forecast (per station)
```python
import statsmodels.api as sm

def sarima_forecast(series, order=(1,0,0), seasonal_order=(1,0,0,8), H=56):
    # series: pd.Series indexed by timestamp (3h) with arrivals
    mod = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    fc = res.get_forecast(steps=H)
    yhat = fc.predicted_mean
    return yhat.rename('yhat_sarima').to_frame()
```

### E. Queueing (Erlang‑C) on future forecasts
```python
def erlang_c_wait(lambda_hour, mu, c):
    rho = lambda_hour / (c * mu)
    if rho >= 1:            # unstable
        return np.inf, rho
    a = lambda_hour / mu    # offered load in Erlangs
    # Erlang‑C probability of wait
    sum_terms = sum((a**k) / math.factorial(k) for k in range(c))
    Pb = (a**c / math.factorial(c)) * (1/(1 - rho))
    Pw = Pb / (sum_terms + Pb)
    # Expected wait in queue (hours), convert to minutes
    Wq_hours = Pw * (1/(c*mu - lambda_hour))
    return Wq_hours*60.0, rho
```

---

## 5) Putting it all together (per station)
1. **Slice** station history → `hist` with columns `timestamp, arrivals`.
2. **Future index**: `future_idx = make_future_bins(hist['timestamp'].max(), H)`.
3. **Run models**:
   - Baselines → `df_base` with `['timestamp','naive','snaive','movavg']`.
   - Prophet → `df_prophet` (C above).
   - SARIMA → `df_sarima` (D above).
   - LSTM → `df_lstm` via (B) using your scaler and last `SEQ` points.
4. **Join** on `timestamp` → `df_fc`.
5. **(Optional) Ensemble**: add `yhat_ens` = weighted blend of available columns.
6. **Queueing**:
   - For each row, compute `lambda_hour = yhat_selected / 3`.
   - Assign `mu` (robust by station & hour‑of‑day), `c` (assets/what‑if).
   - Apply `erlang_c_wait` → add `Wq_min` and `rho` per bin.
7. **Persist**: Save CSV (and/or feed into your Streamlit dashboard).

---

## 6) Validation for future‑use
- **Backtesting**: Simulate “future” by walking forward on the last N days. Refit classical models as needed; for LSTM keep weights fixed and use recursive multi‑step.
- **Horizon‑wise metrics**: MAE/RMSE at h = 1..H to see degradation with horizon.
- **Coverage**: For SARIMA/Prophet, check that actuals fall within forecast intervals at the expected rate.
- **Stability**: Inspect rho across the horizon; flag bins where rho ≥ 1 (unstable).

---

## 7) Common pitfalls & guardrails
- **Scaling drift**: Always invert the scaler for LSTM outputs before clipping to 0, and re‑apply scaling when feeding recursive steps.
- **Seasonality misspecification**: For 3‑hour bins, daily period is 8; weekly is 56. SARIMA’s seasonal order should match the strongest cycle.
- **Calendar leakage**: Do not build future targets from the future! Only features that are known in advance (calendar/holidays) should be used for future inference.
- **Zero‑inflation**: If many zeros remain, consider a hurdle/zero‑inflated variant or a two‑stage model (zero vs non‑zero, then conditional count).
- **Queueing blow‑ups**: Clamp mu>0, c≥1; when rho≥1 treat E[Wq] as infinite or cap at a large sentinel for UI.
- **Ensembling**: If one model dominates on validation, give it more weight; if models disagree wildly, prefer the more conservative route in operations planning.

---

## 8) Recommended next actions
1. Add a **`forecast_future.py`** (or notebook section) implementing the helpers above.
2. Wire a **Streamlit page** to pick horizon **H** and **what‑if** multipliers; render future arrivals + queue waits.
3. Add a **rolling backtest** to auto‑refresh model weights/ensembles monthly.
4. Log **per‑station diagnostics** (utilization, missed peaks) to prioritize capacity upgrades.

---

### Notes
- This plan assumes your existing training artifacts (scaler, LSTM weights, Prophet/SARIMA configs) are available per station.
- If you want, I can turn these snippets into **drop‑in functions** matched exactly to your variable names and file layout.
