@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    orders = Order.query.filter(Order.state.in_(["filled", "partially filled"])).all()
    portfolio = {}

    for order in orders:
        key = (order.security_type, order.symbol)
        if key not in portfolio:
            portfolio[key] = {
                "symbol": order.symbol,
                "security_type": order.security_type,
                "quantity": 0
            }
        portfolio[key]["quantity"] += order.quantity

    return jsonify(list(portfolio.values()))