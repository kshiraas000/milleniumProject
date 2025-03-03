class Order:
    def __init__(self, order_id, symbol, quantity, limit_price, side):
        self.order_id = order_id  # Unique order ID
        self.symbol = symbol  # Stock symbol (e.g., AAPL, TSLA)
        self.quantity = quantity  # Number of shares
        self.limit_price = limit_price  # Max buy price / Min sell price
        self.side = side  # "BUY" or "SELL"
        self.status = "Pending"  # Default status

    def __str__(self):
        return f"Order {self.order_id}: {self.side} {self.quantity} {self.symbol} at ${self.limit_price} - Status: {self.status}"

orders = []  # List to store orders
order_counter = 1  # Auto-increment order ID

def create_order(symbol, quantity, limit_price, side):
    global order_counter
    order = Order(order_counter, symbol, quantity, limit_price, side)
    orders.append(order)
    print(f"✅ Order {order.order_id} created: {side} {quantity} shares of {symbol} at ${limit_price}")
    order_counter += 1  # Increment ID for the next order

def view_orders():
    if not orders:
        print("📭 No orders found.")
        return
    print("\n📋 Current Orders:")
    for order in orders:
        print(order)

def cancel_order(order_id):
    for order in orders:
        if order.order_id == order_id and order.status == "Pending":
            order.status = "Canceled"
            print(f"❌ Order {order_id} canceled.")
            return
    print("⚠️ Order not found or already processed.")

def main():
    while True:
        print("\n===== Order Entry System =====")
        print("1. Create Order")
        print("2. View Orders")
        print("3. Cancel Order")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
            quantity = int(input("Enter quantity: "))
            limit_price = float(input("Enter limit price: "))
            side = input("Enter side (BUY/SELL): ").upper()
            if side not in ["BUY", "SELL"]:
                print("⚠️ Invalid side! Please enter BUY or SELL.")
                continue
            create_order(symbol, quantity, limit_price, side)

        elif choice == "2":
            view_orders()

        elif choice == "3":
            order_id = int(input("Enter order ID to cancel: "))
            cancel_order(order_id)

        elif choice == "4":
            print("🚀 Exiting Order Entry System. Goodbye!")
            break

        else:
            print("⚠️ Invalid choice, please try again.")

if __name__ == "__main__":
    main()
