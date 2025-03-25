from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world!"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
