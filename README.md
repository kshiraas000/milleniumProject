# Millennium Project — Quantitative Trading & Portfolio Simulation Platform  

A full-stack **quantitative trading simulation** system combining order management, portfolio tracking, option analytics, predictive modeling, and AI-powered trade advice.  

This project replicates the **core functions of an institutional trading system** — from order routing and execution to forecasting and strategy recommendations — built entirely with Python, Flask, MySQL, and machine learning.

---

## Project Overview  

The Millennium Project simulates a miniature trading platform that integrates:  

1. **Order Management & Execution Engine**  
2. **Portfolio Tracking & P&L Analysis**  
3. **Options Pricing & Volatility Surface Visualization**  
4. **Predictive Modeling using LSTM Networks**  
5. **AI-driven Order Parsing and Trade Advice (Gemini API)**  

---

## Tech Stack  

**Backend:** Flask (Python), SQLAlchemy, MySQL  
**Frontend:** HTML, CSS, Chart.js  
**Data APIs:** Yahoo Finance (`yfinance`)  
**Machine Learning:** TensorFlow/Keras (LSTM), pandas-ta  
**AI Integration:** Gemini API (Natural Language Processing)  

---

## System Architecture & Data Flow  
User Input → Flask Backend → MySQL Database → Execution Engine
↘ Gemini NLP → Order Parsing
↘ Yahoo Finance → Market Data, Options, Forecasts
↘ LSTM Model → Price Prediction
↘ Chart.js Frontend → Visualization

**Workflow Summary:**  
1. User places an order (manual or via AI command).  
2. Flask backend validates and routes order.  
3. Execution engine simulates fills based on live market data.  
4. Portfolio and P&L values update automatically.  
5. LSTM forecasts future price movement.  
6. Gemini API generates actionable trading advice.  

---

## 1. Order Management & Execution  

### Finance Context  
In financial markets, orders pass through an **Order Management System (OMS)** to be routed to an exchange or broker.  
The system replicates this flow by applying execution rules against live quotes.

**Order Types Supported:**  
- **Market Order:** Execute immediately at best available price.  
- **Limit Order:** Execute only at or better than a specified price.  
- **Stop Order:** Converts to market order when stop price triggered.  
- **Stop-Limit Order:** Becomes a limit order once stop threshold is reached.  

### Technical Implementation  
- **Flask REST API** endpoint `/orders` accepts POST requests with JSON payloads.  
- **MySQL Schema (SQLAlchemy):**

id | symbol | side | qty | type | limit_price | stop_price | status | filled_price | timestamp
- **Execution Logic:**
- Market → Filled at current quote.  
- Limit → Filled only when bid/ask crosses.  
- Stop → Triggered when price hits stop threshold.  

*Simulates a real exchange matching engine locally.*

---

## 2. Portfolio Management & P&L Tracking  

### Finance Context  
A portfolio tracks positions in each security:  
> **Unrealized P&L** = (Market Price − Cost Basis) × Quantity  
> **Realized P&L** = Gains/losses from closed trades  

### Technical Implementation  
- **Positions table** updated dynamically after order execution.  
- **Valuation loop:** Fetches real-time market data via `yfinance`.  
- **Portfolio performance chart:** Aggregates portfolio value over the last 30 days using Chart.js.  

*Core features include cost basis, market value, and total portfolio performance visualization.*

---

## 3. Options Pricing & Volatility Surface  

### Finance Context  
Options derive value from:
- Underlying stock price  
- Strike price  
- Expiration  
- Interest rate  
- Volatility (σ)

**Black-Scholes Model (BSM):**  
\[
C = S_0 N(d_1) - Ke^{-rT} N(d_2)
\]
where implied volatility (IV) is the σ that equates market price with model price.

### Technical Implementation  
- Pulled **option chains** using `yfinance`.  
- Inverted **Black-Scholes** with **Newton’s method** to find IV.  
- Plotted **3D volatility surface** (`strike × expiration × IV`) with Chart.js.  
- Built **option payoff diagrams** for strategy visualization.  

*Visual analytics replicate institutional derivatives dashboards.*

---

## 4. Predictive Modeling (LSTM Network)  

### Finance Context  
Stock prices form time series — exhibiting **autocorrelation, momentum, and trend**.  
LSTM networks model sequential dependencies better than traditional regressions.  

### Technical Implementation  
- Pulled 6 months of OHLCV data via `yfinance`.  
- Generated technical indicators (SMA, EMA, RSI) with `pandas-ta`.  
- Scaled features using `MinMaxScaler`.  
- Built **LSTM model**:  
- Input (timesteps × features)  
- LSTM hidden layers  
- Dense output predicting next-day price  
- Produced **7-day forecast curve** and classified market trend as **bullish/bearish/neutral**.  
- Cached trained models for efficiency.  

*Forecasts guide strategy signals — not as sole trading decisions, reflecting real quant practices.*

---

## 5. AI Order Parsing & Trading Advice  

### Finance Context  
Traders often issue **natural language orders** ("Buy 10 AAPL at 150 if it dips").  
Institutions automate this via NLP and voice-trade interfaces.

### Technical Implementation  
- **Gemini API Integration:**  
- Input: Natural language command  
- Output: JSON structure (`symbol`, `qty`, `type`, `limit_price`)  
- **Trading Advisor:**  
- Combines LSTM forecast + portfolio state  
- Prompts Gemini to generate short strategy advice  

*Example Output:*  
> “AAPL forecast shows +3% over 7 days. Hold long; consider stop at $142.”

*Mimics sell-side research desks delivering concise recommendations.*

---

## 6. Full System Integration  

**Data Sources:**  
- Yahoo Finance (quotes, options, history)  
- Gemini API (NLP + trade advice)  
- MySQL (persistent portfolio + orders)  

**Frontend Visualizations:**  
- Portfolio performance curve  
- Option payoff diagrams  
- Volatility surface  
- Forecast trends  

*All analytics rendered dynamically using Chart.js.*

---

## Tools & Libraries  

| Category | Libraries / Tools |
|-----------|------------------|
| Backend | Flask, SQLAlchemy, MySQL |
| Market Data | yfinance |
| Machine Learning | TensorFlow / Keras, pandas-ta |
| Frontend | HTML, CSS, Chart.js |
| AI Integration | Gemini API |
| Data Processing | pandas, numpy |

---

