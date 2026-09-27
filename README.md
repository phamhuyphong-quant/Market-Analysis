# DCA: Asset Allocation & Investment Comparison (VN Market)

**Question:** If you invest 5,000,000 VND every month from January 2015 to the present, which investment type ends with the highest total asset value?

This is a Data Science coursework project comparing three investment options available to a retail investor in Vietnam:
- **Bank savings** (5%/year fixed interest)
- **DCDS** — Quỹ Đầu Tư Chứng Khoán Năng Động DC, a Vietnamese actively-managed equity fund
- **Gold** and **Silver** (USD-denominated, converted to VND)

It is an earlier, educational project — not the author's main research work. (See [Ứng dụng mô hình học máy xếp hạng...](#) for the flagship independent research project on ML ranking models for the Vietnamese stock market.)

## What it does

1. **Data collection** — pulls DCDS NAV history via `vnstock`, and gold/silver/USD-VND spot prices via `yfinance`.
2. **Cleaning & resampling** — normalizes each series to monthly observations, forward/back-fills gaps, and converts gold/silver into VND per traditional Vietnamese units (chỉ, gram).
3. **Portfolio simulation** — simulates a fixed monthly 5,000,000 VND contribution into each asset type and tracks quantity purchased, leftover cash, and running portfolio value, so the four options can be compared on the same basis.
4. **Feature engineering** — for each asset, builds multi-horizon return (3/6/12-month), moving average, volatility, rolling high/low, and momentum features from the price history.
5. **Prediction** — trains a `RandomForestClassifier` (with SMOTE for class balance) per asset to classify whether the price will be higher 12 months out, using the engineered features.

## Stack

`pandas`, `numpy`, `matplotlib`, `vnstock`, `yfinance`, `scikit-learn`, `imbalanced-learn` (SMOTE)

## Notes & limitations

- This is a **learning/coursework project**, not a validated trading system. The classifier's reported accuracy reflects a small, non-stationary financial time series with limited out-of-sample testing — it should not be read as a reliable forecast of future asset prices.
- No transaction costs, taxes, or liquidity constraints are modeled.
- The notebook's commentary is in Vietnamese; an English summary is provided above for context.
- Historical data is fetched live via `vnstock`/`yfinance` at runtime and is not redistributed in this repo.

## Author

Phạm Huy Phong — built as part of an introductory Data Science course, later used as a stepping stone toward more rigorous quantitative finance work.
