from flask import Flask, request, jsonify
from model import predict_price

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    price = predict_price(features)
    return jsonify({'predicted_price': price})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
