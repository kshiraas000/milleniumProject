from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
# we need to input our MYSQL database here
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Stonegate123!@localhost:3306/order_entry_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)

@app.route('/')
def home():
    return "Hello world!"

# Defining the Order model with basic fields
class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_type = db.Column(db.String(10), nullable=False)  # "buy" or "sell"
    symbol = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    state = db.Column(db.String(20), nullable=False, default='new')  # e.g., new, sent, filled, cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "order_type": self.order_type,
            "symbol": self.symbol,
            "quantity": self.quantity,
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

    new_order = Order(
        order_type=data['order_type'],
        symbol=data['symbol'],
        quantity=data['quantity'],
        state=data.get('state', 'new')
    )
    db.session.add(new_order)
    db.session.commit()
    return jsonify(new_order.to_dict()), 201

# Endpoint to view all orders
@app.route('/orders', methods=['GET'])
def get_orders():
    orders = Order.query.all()
    return jsonify([order.to_dict() for order in orders]), 200

if __name__ == '__main__':
    # Create the database tables 
    with app.app_context():
        db.create_all()
    app.run(debug=True)
