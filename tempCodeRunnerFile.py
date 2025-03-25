if data['order_type'] == 'limit':
    #     if 'limit_price' not in data or data['limit_price'] <= 0:
    #         return jsonify({"error": "Limit price required for limit orders"}), 400