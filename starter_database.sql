CREATE TABLE orders (
  id INT AUTO_INCREMENT PRIMARY KEY,
  order_type VARCHAR(10) NOT NULL,
  symbol VARCHAR(10) NOT NULL,
  quantity INT NOT NULL,
  state VARCHAR(20) NOT NULL DEFAULT 'new',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
