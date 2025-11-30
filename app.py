from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load files (use your full paths if needed)
model = joblib.load('mental_model.pkl')
vectorizer = joblib.load('mental_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get all inputs
    input1 = request.form.get('input1', '')
    input2 = request.form.get('input2', '')
    input3 = request.form.get('input3', '')
    
    # Combine as space-separated single string (just words, no sentence)
    combined_text = f"{input1} {input2} {input3}".strip()
    
    # Vectorize and predict
    vectorized_input = vectorizer.transform([combined_text])
    prediction = model.predict(vectorized_input)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return f"<h2>Prediction: {predicted_label}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
