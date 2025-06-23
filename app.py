import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2
import base64
from datetime import datetime
import json
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk styling yang lebih baik
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi status sesi
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

GARBAGE_CLASSES = {
    'battery': {
        'name': 'Baterai',
        'color': '#5D5D5D', 
        'disposal': 'Daur ulang di tempat pengumpulan yang ditentukan',
        'icon': '🔋'
    },
    'biological': {
        'name': 'Limbah Biologis',
        'color': '#6B8E23', 
        'disposal': 'Buat kompos atau buang di tempat sampah organik',
        'icon': '🍎'
    },
    'brown-glass': {
        'name': 'Kaca Coklat',
        'color': '#8B4513', 
        'disposal': 'Bilas dan pisahkan berdasarkan warna',
        'icon': '🍾'
    },
    'cardboard': {
        'name': 'Kardus',
        'color': '#DEB887',
        'disposal': 'Bersihkan dan pipihkan sebelum didaur ulang',
        'icon': '📦'
    },
    'clothes': {
        'name': 'Pakaian',
        'color': '#D2691E', 
        'disposal': 'Sumbangkan jika masih bisa digunakan, jika tidak buang di tempat daur ulang tekstil',
        'icon': '👕'
    },
    'green-glass': {
        'name': 'Kaca Hijau',
        'color': '#228B22', 
        'disposal': 'Bilas dan pisahkan berdasarkan warna',
        'icon': '🥂'
    },
    'metal': {
        'name': 'Logam',
        'color': '#696969',
        'disposal': 'Bersihkan dan pisahkan logam besi/non-besi',
        'icon': '🔩'
    },
    'paper': {
        'name': 'Kertas',
        'color': '#FFF8DC',
        'disposal': 'Jaga agar tetap kering dan bersih',
        'icon': '📄'
    },
    'plastic': {
        'name': 'Plastik',
        'color': '#FF6347',
        'disposal': 'Periksa kode daur ulang dan bersihkan',
        'icon': '🥤'
    },
    'shoes': {
        'name': 'Sepatu',
        'color': '#A0522D', 
        'disposal': 'Sumbangkan jika masih bisa digunakan, jika tidak buang di tempat daur ulang tekstil/sepatu',
        'icon': '👟'
    },
    'trash': {
        'name': 'Sampah Umum',
        'color': '#2F4F4F',
        'disposal': 'Buang di tempat sampah umum',
        'icon': '🗑️'
    },
    'white-glass': {
        'name': 'Kaca Bening',
        'color': '#F5F5DC', 
        'disposal': 'Bilas dan pisahkan berdasarkan warna',
        'icon': '🥛'
    }
}

# Fungsi memuat model dengan caching
@st.cache_resource
def load_classification_model():
    """Muat model klasifikasi sampah yang telah dilatih"""
    try:
        # Coba muat model
        model = load_model('best_model.keras')
    
        # Coba dapatkan informasi kelas dari model
        if hasattr(model, 'class_indices'):
            st.write(f"Indeks kelas model: {model.class_indices}")
        
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.info("Pastikan file model 'best_model.keras' berada di direktori yang sama dengan skrip ini.")
        return None

def preprocess_image(img, target_size=(224, 224)):
    """Praproses gambar untuk prediksi model"""
    try:
        # Konversi gambar PIL ke array
        if isinstance(img, Image.Image):
            img_array = np.array(img.convert('RGB'))
        else:
            img_array = img
        
        # Ubah ukuran gambar
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalisasi nilai piksel
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Tambahkan dimensi batch
        img_batch = np.expand_dims(img_normalized, axis=0)
    
        return img_batch
    except Exception as e:
        st.error(f"Terjadi kesalahan saat praproses gambar: {str(e)}")
        return None

def predict_garbage_class(model, img):
    """Prediksi kelas sampah dari gambar"""
    try:
        # Praproses gambar
        processed_img = preprocess_image(img)
        if processed_img is None:
            return None, None
        
        # Lakukan prediksi
        predictions = model.predict(processed_img, verbose=0)
        probabilities = predictions[0]
        
        # Informasi debug (dapat dihapus di produksi)
        # st.write(f"Bentuk output model: {predictions.shape}")
        # st.write(f"Jumlah prediksi: {len(probabilities)}")
        # st.write(f"Jumlah kelas yang ditentukan: {len(GARBAGE_CLASSES)}")
        
        # Dapatkan nama kelas dari model atau gunakan kelas yang ditentukan
        class_names = list(GARBAGE_CLASSES.keys())
        
        # Periksa apakah dimensi cocok
        if len(probabilities) != len(class_names):
            st.error(f"Ketidaksesuaian: Model memprediksi {len(probabilities)} kelas, tetapi {len(class_names)} kelas ditentukan")
            
            # Coba dapatkan nama kelas sebenarnya dari model jika tersedia
            if hasattr(model, 'class_names'):
                actual_class_names = model.class_names
                st.write(f"Nama kelas model: {actual_class_names}")
            
            # Buat pemetaan berdasarkan prediksi yang tersedia
            if len(probabilities) < len(class_names):
                class_names = class_names[:len(probabilities)]
                st.warning(f"Menggunakan {len(probabilities)} kelas pertama: {class_names}")
            else:
                # Tambahkan dengan nama generik jika diperlukan
                while len(class_names) < len(probabilities):
                    class_names.append(f"kelas_{len(class_names)}")
                st.warning(f"Nama kelas diperluas: {class_names}")
        
        # Buat kamus hasil
        results = {}
        for i, class_name in enumerate(class_names):
            if i < len(probabilities):
                results[class_name] = float(probabilities[i])
        
        # Dapatkan prediksi teratas
        max_idx = np.argmax(probabilities)
        if max_idx < len(class_names):
            predicted_class = class_names[max_idx]
            confidence = float(probabilities[max_idx])
        else:
            st.error(f"Indeks prediksi {max_idx} di luar jangkauan untuk nama kelas")
            return None, None
        
        return predicted_class, results
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat prediksi: {str(e)}")
        st.write(f"Detail pengecualian: {type(e).__name__}: {str(e)}")
        return None, None

def create_confidence_chart(results):
    """Buat bagan Confidence untuk prediksi"""
    if not results:
        return None
    
    # Siapkan data untuk bagan
    classes = []
    confidences = []
    colors = []
    
    for class_name, confidence in results.items():
        # Hanya gunakan kelas yang ada di GARBAGE_CLASSES
        if class_name in GARBAGE_CLASSES:
            classes.append(GARBAGE_CLASSES[class_name]['name'])
            confidences.append(confidence * 100)
            colors.append(GARBAGE_CLASSES[class_name]['color'])
        else:
            # Untuk kelas yang tidak dikenal, gunakan styling generik
            classes.append(class_name.title())
            confidences.append(confidence * 100)
            colors.append('#888888')
    
    if not classes:  # Tidak ada kelas yang valid ditemukan
        return None
    
    # Buat bagan batang
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=confidences,
            marker_color=colors,
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Prediksi berdasarkan Kelas",
        xaxis_title="Kelas Sampah",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        template="plotly_white"
    )
    
    return fig

def add_to_history(image_data, predicted_class, confidence, results):
    """Tambahkan prediksi ke riwayat"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'results': results
    }
    st.session_state.prediction_history.append(history_entry)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Sistem Klasifikasi Sampah AI</h1>
        <p>Unggah gambar untuk mengklasifikasikan jenis sampah dan dapatkan rekomendasi daur ulang</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Muat model
    model = load_classification_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:        
        st.markdown("# ♻️ Jenis Sampah")
        for class_key, class_info in GARBAGE_CLASSES.items():
            st.markdown(f"""
            **{class_info['icon']} {class_info['name']}**
            """)
        
        # Tombol bersihkan riwayat
        if st.button("🗑️ Bersihkan Riwayat"):
            st.session_state.prediction_history = []
            st.success("Riwayat telah dibersihkan!")
    
    # Konten utama
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Unggah Gambar")
        
        # Pengunggah file
        uploaded_file = st.file_uploader(
            "Pilih gambar...",
            type=['jpg', 'jpeg', 'png'],
            help="Unggah gambar sampah untuk diklasifikasikan"
        )
        
        # Input kamera
        camera_image = st.camera_input("📷 Atau ambil foto")
        
        # Gunakan gambar kamera jika tersedia, jika tidak gunakan file yang diunggah
        image_source = camera_image if camera_image is not None else uploaded_file
        
        if image_source is not None:
            # Tampilkan gambar
            image = Image.open(image_source)
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
            
            # Tombol prediksi
            if st.button("🔍 Analisis Gambar", type="primary"):
                with st.spinner("Menganalisis gambar..."):
                    # Lakukan prediksi
                    predicted_class, results = predict_garbage_class(model, image)
                    
                    if predicted_class and results:
                        confidence = results[predicted_class]
                        class_info = GARBAGE_CLASSES[predicted_class]
                        
                        # Tampilkan hasil prediksi
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2>{class_info['icon']} {class_info['name']}</h2>
                            <h3>Confidence: {confidence*100:.1f}%</h3>
                            <p>{class_info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Rekomendasi pembuangan
                        st.markdown("### 💡 Rekomendasi Pembuangan")
                        
                        if predicted_class == 'trash':
                            st.markdown("""
                            <div class="warning-box">
                                <strong>⚠️ Sampah Umum</strong><br>
                                Item ini harus dibuang di tempat sampah umum.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="info-box">
                                <strong>♻️ Barang Daur Ulang</strong><br>
                                {class_info['disposal']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Tambahkan ke riwayat
                        add_to_history(None, predicted_class, confidence, results)
                        
                        # Simpan hasil dalam status sesi untuk bagan
                        st.session_state.latest_results = results
    
    with col2:
        st.markdown("### 📊 Hasil Analisis")
        
        # Tampilkan bagan Confidence jika hasil tersedia
        if hasattr(st.session_state, 'latest_results'):
            fig = create_confidence_chart(st.session_state.latest_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unggah gambar dan klik 'Analisis Gambar' untuk melihat hasilnya")
        
        # Tampilkan metrik
        if st.session_state.prediction_history:
            st.markdown("### 📈 Session tatistic")
            
            # Hitung statistik
            total_predictions = len(st.session_state.prediction_history)
            avg_confidence = np.mean([entry['confidence'] for entry in st.session_state.prediction_history])
            
            # Kelas paling umum
            classes_predicted = [entry['predicted_class'] for entry in st.session_state.prediction_history]
            most_common_class = max(set(classes_predicted), key=classes_predicted.count) if classes_predicted else None
            
            # Tampilkan metrik
            st.metric("Total Prediksi", total_predictions)
            st.metric("Rata-rata Confidence", f"{avg_confidence*100:.1f}%")
            if most_common_class:
                st.metric("Kelas Paling Umum", GARBAGE_CLASSES[most_common_class]['name'])
    
    # Riwayat Prediksi
    if st.session_state.prediction_history:
        st.markdown("### 📋 Riwayat Prediksi")
        
        # Buat dataframe riwayat
        history_data = []
        for entry in st.session_state.prediction_history[-10:]:  # Tampilkan 10 prediksi terakhir
            history_data.append({
                'Waktu': entry['timestamp'],
                'Kelas Diprediksi': GARBAGE_CLASSES[entry['predicted_class']]['name'],
                'Confidence': f"{entry['confidence']*100:.1f}%",
                'Ikon': GARBAGE_CLASSES[entry['predicted_class']]['icon']
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #white; padding: 1rem;">
        <p>🌱 Bantu lindungi lingkungan dengan klasifikasi dan daur ulang sampah yang tepat</p>
        <p>Dibuat oleh Kelompok 6 Machine Learning | 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()