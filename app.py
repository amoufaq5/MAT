from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/multimodal_model.h5')

# Dummy tokenizer – replace with your actual tokenizer (e.g., using Keras Tokenizer or a transformer)
def tokenize_text(text, max_length=100):
    # For demonstration: convert each character to an int mod vocab size (placeholder)
    tokens = [ord(c) % 5000 for c in text][:max_length]
    tokens = pad_sequences([tokens], maxlen=max_length)
    return tokens

def preprocess_image(image_bytes, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # remove alpha channel if present
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    # Expecting multipart/form-data with a 'text' field and an 'image' file
    data = request.form
    text_data = data.get('text', '')
    image_file = request.files.get('image', None)
    
    if not text_data or image_file is None:
        return jsonify({'error': 'Both text and image inputs are required.'}), 400
    
    # Preprocess text and image inputs
    text_input = tokenize_text(text_data)
    image_input = preprocess_image(image_file.read())

    # Predict diagnosis
    preds = model.predict([text_input, image_input])
    diagnosis_idx = np.argmax(preds, axis=1)[0]
    
    # Map the predicted index to disease names (update mapping as needed)
    disease_mapping = {
        0: 'Common Cold',
        1: 'Pneumonia',
        2: 'Bronchitis',
        3: 'Influenza',
        4: 'Herpes',
        5: 'COVID-19',
        6: 'Hepatitis B',
        7: 'Chlamydia',
        8: 'Chicken Pox',
        9: 'Other'
    }
    diagnosis = disease_mapping.get(diagnosis_idx, 'Unknown')
    
    # Decide action based on model confidence and diagnosis – for demo, a simple threshold is used
    confidence = float(np.max(preds))
    if confidence < 0.5:
        action = "Refer to doctor"
    else:
        action = "Prescribe OTC" if diagnosis in ['Common Cold', 'Influenza', 'Chicken Pox'] else "Refer to doctor"
    
    return jsonify({
        'diagnosis': diagnosis,
        'confidence': confidence,
        'action': action
    })

if __name__ == '__main__':
    app.run(debug=True)
