from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from random import choice

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Stonegate123!@127.0.0.1:3306/order_entry_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True


db = SQLAlchemy(app)

@app.route('/')
def home():
    return "Hello world!"

# Defining the Order model with basic fields
class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    order_type = db.Column(db.String(20), nullable=False)  # market, limit, stop, stop-limit, trailing-stop
    symbol = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    limit_price = db.Column(db.Float, nullable=True)   # for limit & stop-limit
    stop_price = db.Column(db.Float, nullable=True)    # for stop & stop-limit
    trail_amount = db.Column(db.Float, nullable=True)  # for trailing stop
    state = db.Column(db.String(20), nullable=False, default='new')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "order_type": self.order_type,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "trail_amount": self.trail_amount,
            "state": self.state,
            "created_at": self.created_at.isoformat()
        }

# Endpoint to enter a new order
@app.route('/orders', methods=['POST'])
def create_order():
    data = request.get_json()
    # Basic validation for required fields
    if not data or 'order_type' not in data or 'symbol' not in data or 'quantity' not in data:
        return jsonify({"error": "Missing order data"}), 400

     # Validate limit orders
    # if data['order_type'] == 'limit':
    #     if 'limit_price' not in data or data['limit_price'] <= 0:
    #         return jsonify({"error": "Limit price required for limit orders"}), 400

    new_order = Order(
        order_type=data['order_type'],
        symbol=data['symbol'],
        quantity=data['quantity'],
        limit_price=data.get('limit_price'),
        stop_price=data.get('stop_price'),
        trail_amount=data.get('trail_amount'),
        state=data.get('state', 'new')
        )
    db.session.add(new_order)
    db.session.commit()
    return jsonify(new_order.to_dict()), 201

@app.route('/orders/<int:order_id>/execute', methods=['PUT'])
def execute_order(order_id):
    order = Order.query.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404

    if order.state not in ["new", "sent"]:
        return jsonify({"error": "Order cannot be executed from this state"}), 400

    # Simulating execution with a random state change
    execution_state = choice(["filled", "partially filled"])
    order.state = execution_state
    db.session.commit()
    return jsonify({"message": f"Order {order_id} has been {execution_state}"}), 200

@app.route('/orders/<int:order_id>/state', methods=['PUT'])
def change_order_state(order_id):
    order = Order.query.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404

    data = request.get_json()
    new_state = data.get('state')

    # Valid states
    valid_states = ["new", "sent", "filled", "partially filled", "canceled", "rejected"]

    if new_state not in valid_states:
        return jsonify({"error": f"Invalid state. Valid states: {valid_states}"}), 400

    order.state = new_state
    db.session.commit()
    return jsonify({"message": f"Order {order_id} updated to {new_state}"}), 200

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

if __name__ == '__main__':
    # Create the database tables 
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)


