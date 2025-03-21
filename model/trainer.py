import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Conv2D, MaxPooling2D, Flatten
import numpy as np

# Text branch: processes symptom/question text
def build_text_model(vocab_size, embedding_dim, max_length):
    text_input = Input(shape=(max_length,), name='text_input')
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(text_input)
    x = LSTM(64)(x)
    return Model(inputs=text_input, outputs=x)

# Image branch: processes CT/MRI scans or other images
def build_image_model(input_shape):
    image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(inputs=image_input, outputs=x)

# Combined multi-modal model
def build_multimodal_model(vocab_size, embedding_dim, max_length, image_shape, num_classes):
    text_model = build_text_model(vocab_size, embedding_dim, max_length)
    image_model = build_image_model(image_shape)
    
    # Combine features from both modalities
    combined_features = Concatenate()([text_model.output, image_model.output])
    x = Dense(128, activation='relu')(combined_features)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[text_model.input, image_model.input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Set parameters – adjust these based on your actual data
    vocab_size = 5000
    embedding_dim = 128
    max_length = 100
    image_shape = (128, 128, 3)
    num_classes = 10  # e.g., common cold, pneumonia, bronchitis, influenza, herpes, covid, hepatitis b, chlamydia, chicken pox, other

    model = build_multimodal_model(vocab_size, embedding_dim, max_length, image_shape, num_classes)
    model.summary()

    # Generate dummy data for demonstration (replace with your real annotated dataset)
    num_samples = 100
    X_text = np.random.randint(1, vocab_size, size=(num_samples, max_length))
    X_image = np.random.random(size=(num_samples, *image_shape))
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, size=(num_samples,)), num_classes)

    # Train the model
    model.fit([X_text, X_image], y, epochs=10, batch_size=8)
    model.save('model/multimodal_model.h5')
    print("Model training complete and saved to model/multimodal_model.h5.")

if __name__ == "__main__":
    main()
