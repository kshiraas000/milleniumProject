from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from random import choice
from threading import Thread
import time
import random
import os
from dotenv import load_dotenv 
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env file
# we need to input our MYSQL database here
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True


db = SQLAlchemy(app)

from flask import send_from_directory

@app.route('/')
def home_page():
    return send_from_directory('.', 'index.html')

@app.route('/options')
def options_page():
    return send_from_directory('.', 'options.html')

@app.route('/test-db')
def test_db():
    return jsonify([o.to_dict() for o in Order.query.all()])

@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    orders = Order.query.filter(Order.state.in_(["filled", "partially filled"])).all()
    portfolio = {}

    for order in orders:
        key = (order.security_type, order.symbol.upper())
        if key not in portfolio:
            portfolio[key] = {
                "symbol": order.symbol.upper(),
                "security_type": order.security_type,
                "quantity": 0,
                "latest_price": 0,
                "position_value": 0
            }
        portfolio[key]["quantity"] += order.quantity

    for key, entry in portfolio.items():
        try:
            symbol = entry["symbol"]
            stock = yf.Ticker(symbol)
            price = stock.history(period="1d", interval="1m")["Close"].iloc[-1]
            entry["latest_price"] = round(price, 2)
            entry["position_value"] = round(price * entry["quantity"], 2)
        except:
            entry["latest_price"] = None
            entry["position_value"] = None

    return jsonify(list(portfolio.values()))

@app.route('/portfolio/opinion/<symbol>', methods=['GET'])
def get_opinion(symbol):
    try:
        # Grab security type from existing orders
        entry = next(
            (order.to_dict() for order in Order.query.filter_by(symbol=symbol.upper()).all()),
            {}
        )

        security_type = entry.get("security_type", "stock")

        prediction_data = predict_stock(symbol).json
        trend = prediction_data.get("trend")
        pct = prediction_data.get("percentage_change")
        price = prediction_data.get("predicted_price")
        current = prediction_data.get("current_price")

        prompt = f"""
        You are a financial advisor. A client is currently holding a {trend} position in a {symbol.upper()} {security_type} security.
        Analyze the following 7-day forecast and provide a recommendation for next steps:
        - Current Price: {current}
        - Predicted Price in 7 Days: {price}
        - Expected Change: {pct}%
        - Trend: {trend}

        If it's a stock, advise whether to hold, sell, or take action (e.g. set stop-limit).
        If it's an option, consider time decay, strike price behavior, or volatility.
        Make the advice clear, brief (1–2 sentences), and professional.
        """

        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return jsonify({"advice": response.text.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Simulated price store (in-memory for now)
# live_prices = {
#     "AAPL": 150.00,
#     "MSFT": 310.00,
#     "TSLA": 720.00,
#     "NVDA": 920.00
# }

# def simulate_price_feed():
#     while True:
#         for symbol in live_prices:
#             # Simulate a small price change
#             change = random.uniform(-0.5, 0.5)
#             live_prices[symbol] = round(max(0, live_prices[symbol] + change), 2)
#         time.sleep(1)  # update every second
@app.route('/dashboard')
def dashboard_page():
    return send_from_directory('.', 'dashboard.html')


def fetch_option_price(symbol, option_type, strike, expiration_date):
    """
    Returns the latest option premium (midpoint, ask, last, or whatever you prefer).
    """
    try:
        ticker = yf.Ticker(symbol)
        # expiration_date should be a string in format YYYY-MM-DD for yfinance
        # If your Order model stored it as a Python date, convert it:
        expiration_str = expiration_date.strftime('%Y-%m-%d')
        
        # This returns a namedtuple: (calls, puts)
        opt_chain = ticker.option_chain(expiration_str)
        
        # Depending on your order’s call/put:
        if option_type.lower() == 'call':
            df = opt_chain.calls
        else:
            df = opt_chain.puts
        
        # Now find the row with the matching strike
        row = df.loc[df['strike'] == float(strike)]
        if row.empty:
            return None
        
        # You can pick from "lastPrice", "ask", "bid", "mark" (midpoint), etc.
        # For example, use the lastPrice as "market_price":
        market_price = row['lastPrice'].iloc[0]
        return float(market_price)

    except Exception as e:
        print("Error fetching option price:", e)
        return None



@app.route('/price/<symbol>')
def get_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        # 1-minute resolution is the finest granularity yfinance supports
        df = stock.history(period="1d", interval="1m") 
        if df.empty:
            return jsonify({"error": "Symbol not found"}), 404
        
        # Grab the most recent close (i.e. last row)
        price = df["Close"].iloc[-1]
        return jsonify({"symbol": symbol.upper(), "price": round(price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Defining the Order model with basic fields
class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    security_type = db.Column(db.String(10), default='stock')
    order_type = db.Column(db.String(20), nullable=False)
    order_category = db.Column(db.String(20), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

    option_type = db.Column(db.String(4), nullable=True)
    strike_price = db.Column(db.Float, nullable=True)
    expiration_date = db.Column(db.Date, nullable=True)

    limit_price = db.Column(db.Float, nullable=True)
    stop_price = db.Column(db.Float, nullable=True)
    trail_amount = db.Column(db.Float, nullable=True)

    state = db.Column(db.String(20), nullable=False, default='new')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    execution_price = db.Column(db.Float, nullable=True)
    execution_time = db.Column(db.DateTime, nullable=True)


    def to_dict(self):
        return {
            "id": self.id,
            "security_type": self.security_type,
            "order_type": self.order_type,
            "order_category": self.order_category,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "option_type": self.option_type,
            "strike_price": self.strike_price,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "trail_amount": self.trail_amount,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "execution_price": self.execution_price,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None
        }

# Endpoint to enter a new order
@app.route('/orders', methods=['POST'])
def create_order():

    data = request.get_json()
    # Basic validation for required fields
    if not data or 'order_type' not in data or 'symbol' not in data or 'quantity' not in data:
        return jsonify({"error": "Missing order data"}), 400

    expiration_str = data.get('expiration_date')
    expiration_date = datetime.strptime(expiration_str, "%Y-%m-%d").date() if expiration_str else None

    new_order = Order(
        security_type=data.get('security_type', 'stock'),
        order_type=data['order_type'],
        order_category=data['order_category'],
        symbol=data['symbol'],
        quantity=data['quantity'],
        limit_price=data.get('limit_price'),
        stop_price=data.get('stop_price'),
        trail_amount=data.get('trail_amount'),
        option_type=data.get('option_type'),
        strike_price=data.get('strike_price'),
        expiration_date=expiration_date,
        state=data.get('state', 'new'),
        # execution_price = db.Column(db.Float, nullable=True),
        # execution_time = db.Column(db.DateTime, nullable=True)
    )

    db.session.add(new_order)
    db.session.commit()
    return jsonify(new_order.to_dict()), 201

import traceback

@app.route('/orders/<int:order_id>/execute', methods=['PUT'])
def execute_order(order_id):
    try:
        order = Order.query.get(order_id)
        if not order:
            return jsonify({"error": "Order not found"}), 404

        print(f"🔍 Order fetched: {order.to_dict()}")

        if order.state not in ["new", "sent"]:
            print("🚫 Order state is not executable")
            return jsonify({"error": "Order cannot be executed from this state"}), 400
        try:
            stock = yf.Ticker(order.symbol)
            df = stock.history(period="1d", interval="1m")
            if df.empty:
                return jsonify({"error": "Symbol not found"}), 404
            market_price = df["Close"].iloc[-1]
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        execution_price = round(market_price, 2)
        filled = False

        if order.order_category == "market":
            filled = True
            order.state = "filled"

        elif order.order_category == "limit":
            if order.order_type == "buy" and market_price <= order.limit_price:
                filled = True
                order.state = "filled"
            elif order.order_type == "sell" and market_price >= order.limit_price:
                filled = True
                order.state = "filled"

        elif order.order_category == "stop":
            if order.order_type == "buy" and market_price >= order.stop_price:
                filled = True
                order.state = "filled"
            elif order.order_type == "sell" and market_price <= order.stop_price:
                filled = True
                order.state = "filled"

        elif order.order_category == "stop-limit":
            # Stop condition met?
            if order.order_type == "buy" and market_price >= order.stop_price:
                if market_price <= order.limit_price:
                    filled = True
                    order.state = "filled"
            elif order.order_type == "sell" and market_price <= order.stop_price:
                if market_price >= order.limit_price:
                    filled = True
                    order.state = "filled"

        elif order.order_category == "trailing-stop":
            if order.trail_amount is not None:
                # Initialize anchor if not yet set
                if not hasattr(order, 'trailing_anchor') or order.trailing_anchor is None:
                    order.trailing_anchor = market_price
                else:
                    if order.order_type == "sell":
                        if market_price > order.trailing_anchor:
                            order.trailing_anchor = market_price
                        stop_trigger = order.trailing_anchor - order.trail_amount
                        if market_price <= stop_trigger:
                            filled = True
                            order.state = "filled"
                    elif order.order_type == "buy":
                        if market_price < order.trailing_anchor:
                            order.trailing_anchor = market_price
                        stop_trigger = order.trailing_anchor + order.trail_amount
                        if market_price >= stop_trigger:
                            filled = True
                            order.state = "filled"

        if filled:
            order.execution_price = execution_price
            order.execution_time = datetime.utcnow()
            db.session.commit()
            print(f"✅ Filled at ${execution_price}")
            return jsonify({
                "message": f"Order {order_id} filled.",
                "filled": True,
                "symbol": order.symbol,
                "execution_price": execution_price,
                "timestamp": order.execution_time.isoformat(),
                "state": order.state
            }), 200

        return jsonify({"message": "Order not filled."}), 200

    except Exception as e:
        print("❌ EXCEPTION THROWN DURING EXECUTION:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# @app.route('/orders/<int:order_id>/execute', methods=['PUT'])
# def execute_order(order_id):
#     order = Order.query.get(order_id)
#     if not order:
#         return jsonify({"error": "Order not found"}), 404

#     if order.state not in ["new", "sent"]:
#         return jsonify({"error": "Order cannot be executed from this state"}), 400

    
#     # Fetch live price
#     price_response = get_price(order.symbol)
#     if isinstance(price_response, tuple):
#         price_json = price_response[0].json
#         market_price = price_json.get("price")
#     else:
#         return jsonify({"error": "Unable to fetch live price"}), 500

#     if market_price is None:
#         return jsonify({"error": "Market price unavailable"}), 500

#     filled = False
#     execution_price = round(market_price, 2)
   

#     # 🔥 MARKET ORDER logic
#     if order.order_type == "market":
#         order.state = "filled"
#         order.execution_price = execution_price
#         order.execution_time = datetime.utcnow()
#         db.session.commit()
#         return jsonify({
#             "message": f"Order {order_id} filled.",
#             "filled": True,
#             "symbol": order.symbol,
#             "execution_price": execution_price,
#             "timestamp": order.execution_time.isoformat(),
#             "state": order.state
#         }), 200

#     # fallback (limit/stop/etc not triggered)
#     return jsonify({"message": "Order not filled. Conditions not met."}), 200


@app.route('/orders/<int:order_id>/cancel', methods=['PUT'])
def cancel_order(order_id):
    order = Order.query.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404

    if order.state in ["filled", "partially filled"]:
        return jsonify({"error": "Order cannot be canceled after execution"}), 400

    order.state = "canceled"
    db.session.commit()
    return jsonify({"message": f"Order {order_id} has been canceled"}), 200

@app.route('/orders', methods=['GET'])
def get_orders():
    state_filter = request.args.get('state')
    
    if state_filter:
        orders = Order.query.filter_by(state=state_filter).all()
    else:
        orders = Order.query.all()
    
    return jsonify([order.to_dict() for order in orders]), 200

# In-memory cache to avoid reloading models repeatedly
MODEL_CACHE = {}
SCALER_CACHE = {}

def load_model_and_scaler(symbol):
    symbol = symbol.lower()
    model_path = f"models/{symbol}_lstm.h5"
    scaler_path = f"models/{symbol}_scaler.npy"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model not found for {symbol.upper()} – training now...")
        train_and_save_model(symbol)

    # Always reload after retrain
    MODEL_CACHE.pop(symbol, None)
    SCALER_CACHE.pop(symbol, None)

    if symbol not in MODEL_CACHE:
        MODEL_CACHE[symbol] = load_model(model_path, compile=False)
        SCALER_CACHE[symbol] = np.load(scaler_path, allow_pickle=True).item()

    return MODEL_CACHE[symbol], SCALER_CACHE[symbol]

def train_and_save_model(symbol):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    df = yf.Ticker(symbol).history(period="1y", interval="1d")

    if df.empty or len(df) < 100:
        raise Exception(f"Not enough data to train model for {symbol.upper()}")

    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)

    feature_cols = ['Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI']
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    window = 60
    for i in range(window, len(scaled_data) - 7):
        X.append(scaled_data[i - window:i])
        y.append(scaled_data[i:i + 7, 0])  # only Close

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol.lower()}_lstm.h5", include_optimizer=False)
    np.save(f"models/{symbol.lower()}_scaler.npy", scaler)

@app.route('/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    try:
        # 1. Get the most recent 60 days of data
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo", interval="1d")

        if df.empty or len(df) < 65:
            return jsonify({"error": "Not enough recent data for prediction"}), 404

        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df.dropna(inplace=True)

        # 2. Select only the last 60 days of features
        feature_cols = ['Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI']
        latest_data = df[feature_cols].values[-60:]

        # 3. Load model + scaler
        model, scaler = load_model_and_scaler(symbol)
        scaled_input = scaler.transform(latest_data).reshape(1, 60, len(feature_cols))

        # 4. Make 7-day prediction
        pred_scaled = model.predict(scaled_input)[0]
        pred_combined = np.zeros((7, len(feature_cols)))
        pred_combined[:, 0] = pred_scaled  # Only 'Close' is predicted
        pred_actual = scaler.inverse_transform(pred_combined)[:, 0]

        predictions = [round(float(p), 2) for p in pred_actual]
        current_price = round(float(df['Close'].iloc[-1]), 2)
        predicted_final = predictions[-1]
        expected_change = round(predicted_final - current_price, 2)
        pct_change = round((expected_change / current_price) * 100, 2)

        return jsonify({
            "symbol": symbol.upper(),
            "current_price": current_price,
            "predicted_price": predicted_final,
            "expected_change": expected_change,
            "percentage_change": pct_change,
            "trend": "profit" if expected_change > 0 else "loss",
            "predictions": {f"Day {i+1}": p for i, p in enumerate(predictions)}
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train/<symbol>", methods=["POST"])
def force_train(symbol):
    symbol = symbol.upper()
    try:
        print(f"🔁 Forcing retrain for {symbol}...")

        # Step 1: Retrain model on fresh data
        train_and_save_model(symbol)

        # Step 2: Clear any cached model/scaler in memory
        MODEL_CACHE.pop(symbol.lower(), None)
        SCALER_CACHE.pop(symbol.lower(), None)

        print(f"Retrained and reloaded model for {symbol}")
        return jsonify({"message": f"{symbol} model retrained and reloaded."})

    except Exception as e:
        print(f"Retrain error for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/nlp_order", methods=["POST"])
def parse_natural_order():
    user_input = request.json.get("text")
    if not user_input:
        return jsonify({"error": "No input text provided"}), 400

    prompt = f"""
You are a trading assistant. Convert the following natural language into a JSON object for order placement.
Fields: order_type, symbol, quantity, limit_price, stop_price, trail_amount (if available).

Example:
"I want to buy 10 shares of NVDA if it drops below $850" → 
{{
  "order_type": "limit",
  "symbol": "NVDA",
  "quantity": 10,
  "limit_price": 850
}}

Now convert:
"{user_input}"
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")  # Correct path
        response = model.generate_content(prompt)
        parsed = response.text.strip()

        # Clean up Gemini markdown output
        if parsed.startswith("```json"):
            parsed = parsed.replace("```json", "").replace("```", "").strip()

        return jsonify({"parsed": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create the database tables 
    with app.app_context():
        db.create_all()
    # Thread(target=simulate_price_feed, daemon=True).start()
    #Thread(target=auto_execute_orders, daemon=True).start()
    app.run(debug=True, port=5001)