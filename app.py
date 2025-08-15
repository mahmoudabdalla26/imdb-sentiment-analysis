from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flask app
app = Flask(__name__)

# Load model and tokenizer at startup
MODEL_PATH = r"F:\Projects\Sentiment Analysis\models\gru_tuned_model.h5"
TOKENIZER_PATH = r"F:\Projects\Sentiment Analysis\models\tokenizer.pkl"

model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

MAX_LEN = 200  # same as training

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        review_text = request.form["review"]
        seq = tokenizer.texts_to_sequences([review_text])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        pred = model.predict(pad)[0][0]
        prediction = "Positive" if pred > 0.5 else "Negative"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
