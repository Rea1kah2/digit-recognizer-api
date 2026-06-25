import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import re

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Flask API is running!'})

# Load Model
model = tf.keras.models.load_model('model/mnist_model.h5') # Model akan diload sekali saja, karena kalau berulang kali maka server akan lambat. Jadi kita load model di awal saja.
print("Model berhasil dimuat!")


# Route untuk prediksi digit
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data gambar dari request
        data = request.get_json()
        image_data = data['image']

        # Bersihkan prefix dulu
        image_data = re.sub(r'^data:image/.+;base64,', '', image_data)

        # Decode base64 jadi bytes
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocessing gambar
        image = image.convert('RGBA')
        white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        white_bg.paste(image, mask=image.split()[3]) # Tempel dengan alpha mask

        image = white_bg.convert('L') # ubah ke grayscale
        image = image.resize((28, 28)) # resize ke 28x28

        image_array = np.array(image).astype('float32')

        # invert warna
        image_array = 255 - image_array

        # Reshape sesuai input model (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)

        # Prediksi
        predictions = model.predict(image_array, verbose=0)

        # predictions[0] = array 10 nilai probabilitas untuk angka 0-9
        # Contoh: [0.01, 0.02, 0.01, 0.90, 0.01, ...]
        # Artinya: model 90% yakin ini angka 3

        predicted_digit = int(np.argmax(predictions[0])) # angka dengan probabilitas tertinggi
        confidence = float(np.max(predictions[0])) * 100 # Konversi ke persen

        # Semua probabilitas utk ditampilkan di UI sebagai bar chart

        all_probs = {
            str(i): round(float(predictions[0][i]) * 100, 2)
            for i in range(10)
        }

        # Return hasilnya
        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

# Jalankan server
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"\nServer berjalan di http://localhost:{port}")
    app.run(debug=True, port=port)


