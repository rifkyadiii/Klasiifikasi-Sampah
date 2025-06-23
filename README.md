# Proyek Klasifikasi Sampah dengan Deep Learning (Xception Transfer Learning)

## Gambaran Umum Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi sampah otomatis menggunakan teknik Deep Learning, khususnya Transfer Learning dengan arsitektur Xception. Tujuannya adalah untuk mengidentifikasi berbagai jenis sampah dari gambar, yang dapat membantu dalam upaya daur ulang dan pengelolaan limbah yang lebih efisien.

Dataset yang digunakan terdiri dari 12 kategori sampah yang berbeda: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, dan `white-glass`.

## Dataset

Dataset "Garbage Classification" diunduh dari Kaggle.

  * **Sumber:** [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
  * **Jumlah Kelas:** 12
  * **Total Gambar:** 15515 gambar.

### Struktur Dataset setelah Splitting

Dataset dibagi menjadi tiga set:

  * **Training Set:** 60% dari data asli (digunakan untuk melatih model).
  * **Validation Set:** 20% dari data asli (digunakan untuk memantau kinerja model selama pelatihan dan mencegah *overfitting*).
  * **Testing Set:** 20% dari data asli (digunakan untuk evaluasi akhir model yang belum pernah dilihat sebelumnya).

Distribusi gambar per kelas:

![image](https://github.com/user-attachments/assets/dba7cd52-b5f8-4b53-bdb5-79c85d7e2caa)

  * `battery`: 400 gambar
  * `Kelas metal`: 769 gambar
  * `Kelas white-glass`: 775 gambar
  * `Kelas biological`: 985 gambar
  * `Kelas paper`: 1050 gambar
  * `Kelas brown-glass`: 607 gambar
  * `Kelas battery`: 945 gambar
  * `Kelas trash`: 697 gambar
  * `Kelas cardboard`: 891 gambar
  * `Kelas shoes`: 1977 gambar
  * `Kelas clothes`: 5325 gambar
  * `Kelas plastic`: 865 gambar
  * `Kelas green-glass`: 629 gambar

## Arsitektur Model

Model menggunakan pendekatan **Transfer Learning** dengan `Xception` sebagai *base model*. Arsitektur `Xception` dikenal efektif untuk tugas klasifikasi gambar.

### Detail Model:

  * **Base Model:** Xception (dilatih pada dataset ImageNet).
  * **Lapisan Beku (Frozen Layers):** Lapisan awal Xception dibekukan untuk memanfaatkan fitur yang telah dipelajari.
  * **Fine-tuning:** Beberapa lapisan terakhir (`fine_tune_at = -20`) dari base model `Xception` tidak dibekukan dan dilatih ulang dengan learning rate yang sangat rendah. Ini memungkinkan model untuk menyesuaikan fitur yang lebih spesifik untuk dataset sampah.
  * **Custom Head:** Ditambahkan lapisan kustom (dense layers) di atas base model:
      * `GlobalAveragePooling2D()`: Merampingkan fitur.
      * `Dense(256, activation='relu', kernel_regularizer=l2(0.01))`: Lapisan Dense dengan 256 unit, aktivasi ReLU, dan regulasi L2 yang kuat untuk mencegah overfitting.
      * `BatchNormalization()`: Menstabilkan pelatihan.
      * `Dropout(0.6)`: Dropout rate tinggi (60%) untuk mengurangi overfitting.
      * `Dense(128, activation='relu', kernel_regularizer=l2(0.01))`: Lapisan Dense dengan 128 unit, aktivasi ReLU, dan regulasi L2.
      * `BatchNormalization()`: Menstabilkan pelatihan.
      * `Dropout(0.5)`: Dropout rate tinggi (50%).
      * `Dense(num_classes, activation='softmax')`: Lapisan output dengan aktivasi softmax untuk klasifikasi multi-kelas.

### Strategi Pencegahan Overfitting:

  * **Data Augmentation:** Digunakan augmentasi data konservatif (`rotation_range=20`, `width_shift_range=0.1`, `zoom_range=0.1`, dll.) untuk meningkatkan variasi data training.
  * **Regularisasi L2:** Diterapkan pada lapisan Dense kustom.
  * **Dropout:** Digunakan dropout rate tinggi (0.6 dan 0.5) setelah lapisan Dense.
  * **Batch Normalization:** Membantu menstabilkan pelatihan dan mengurangi ketergantungan pada inisialisasi awal.
  * **Learning Rate Rendah:** Optimizer Adam dengan `learning_rate=1e-5` (sangat rendah) untuk fine-tuning yang hati-hati.
  * **Early Stopping:** Menghentikan pelatihan jika `val_loss` tidak membaik setelah sejumlah epoch (`patience=10`).
  * **Model Checkpoint:** Menyimpan bobot model terbaik berdasarkan `val_accuracy`.
  * **ReduceLROnPlateau:** Mengurangi learning rate jika `val_loss` tidak membaik, membantu model keluar dari local minima.
  * **Peningkatan Ukuran Batch:** `BATCH_SIZE = 32` membantu generalisasi yang lebih baik.
  * **Pengurangan Ukuran Input:** `IMAGE_SIZE = 224` untuk mengurangi kompleksitas model.
  * **Pengurangan Jumlah Epoch:** `EPOCHS = 50` sebagai batas atas.

## Hasil Pelatihan dan Evaluasi Model

Model dilatih selama 50 epoch (atau dihentikan lebih awal oleh Early Stopping jika kriteria terpenuhi). Setelah pelatihan, model dievaluasi pada tiga set data yang berbeda: Training, Validation, dan Testing untuk mendapatkan gambaran komprehensif tentang kinerjanya.

Berikut adalah ringkasan kinerja model setelah pelatihan:

  * **Akurasi Training Akhir:** Akurasi model pada data yang digunakan untuk pelatihan. Nilai tinggi (\~95.49%) menunjukkan bahwa model telah belajar dengan baik dari data training.
  * **Akurasi Validasi Akhir:** Akurasi model pada data yang tidak terlihat selama pelatihan tetapi digunakan untuk memantau kemajuan dan mencegah *overfitting*. Akurasi validasi yang tinggi (\~96.26%) dan dekat dengan akurasi training menunjukkan generalisasi yang baik.
  * **Akurasi Testing:** Akurasi model pada data yang benar-benar baru, belum pernah dilihat oleh model selama pelatihan maupun validasi. Ini adalah metrik paling objektif untuk mengukur seberapa baik model akan berkinerja di dunia nyata. Akurasi testing yang tinggi (\~96.25%) mengindikasikan model sangat efektif dalam mengklasifikasikan jenis sampah baru.
  * **Loss Testing:** Nilai *loss* model pada data testing. Nilai rendah (\~0.1509) menunjukkan bahwa prediksi model sangat akurat dan memiliki kesalahan yang minimal pada data yang belum pernah dilihat.

### Visualisasi Hasil Training

![image](https://github.com/user-attachments/assets/03969419-9683-4aef-b35f-f7ac43171f3e)

  * **Model Accuracy & Loss:** Grafik ini menunjukkan bagaimana akurasi dan loss berubah seiring dengan bertambahnya epoch, baik untuk data training maupun validasi. Konvergensi yang baik dengan *gap* yang kecil antara kedua kurva menunjukkan model belajar dengan baik dan tidak terlalu *overfit*.
  * **Accuracy Gap Analysis & Loss Gap Analysis:** Grafik ini memvisualisasikan perbedaan (gap) antara metrik training dan validasi. Gap yang kecil dan stabil di bawah ambang batas (garis putus-putus oranye dan merah) adalah indikator kuat bahwa strategi regulasi (dropout, L2, dll.) bekerja efektif dalam mencegah *overfitting*.
  * **Final Performance Comparison:** Bagan batang ini secara langsung membandingkan akurasi akhir model pada set Training, Validation, dan Testing. Akurasi yang konsisten tinggi di ketiga set adalah tanda kinerja model yang kuat dan kemampuan generalisasi yang sangat baik. Bahkan, akurasi validasi sedikit lebih tinggi dari training, yang bisa menjadi tanda bahwa model memiliki kemampuan generalisasi yang sangat baik dan/atau Early Stopping berhasil menangkap bobot terbaik.
  * **Generalization Analysis:** Grafik ini lebih lanjut menganalisis *gap* antara Train-Val dan Val-Test. Gap yang sangat kecil (misalnya, -0.77% dan 0.41%) menegaskan kemampuan generalisasi model yang kuat, yang berarti model tidak hanya menghafal data pelatihan tetapi juga dapat membuat prediksi akurat pada data baru.

Secara keseluruhan, metrik evaluasi dan analisis visual menunjukkan bahwa model ini terlatih dengan baik, mampu menggeneralisasi dengan efektif, dan siap untuk digunakan dalam aplikasi praktis.

## Deployment (Streamlit)

Model yang telah dilatih (`best_model.keras`) akan di-deploy sebagai aplikasi web interaktif menggunakan Streamlit. Aplikasi ini memungkinkan pengguna untuk:

1.  **Mengunggah Gambar:** Pengguna dapat mengunggah gambar sampah dari perangkat mereka.
2.  **Mengambil Foto:** Pengguna juga dapat mengambil foto langsung menggunakan kamera perangkat mereka.
3.  **Klasifikasi Otomatis:** Model akan memproses gambar dan memprediksi jenis sampah dengan tingkat kepercayaan.
4.  **Rekomendasi Pembuangan:** Aplikasi akan memberikan rekomendasi tentang cara membuang atau mendaur ulang sampah yang terdeteksi.
5.  **Riwayat Prediksi:** Menyimpan dan menampilkan riwayat prediksi dalam sesi saat ini.
6.  **Statistik Sesi:** Menampilkan metrik seperti total prediksi dan rata-rata kepercayaan.
7.  **Visualisasi Kepercayaan:** Menampilkan bagan batang yang menunjukkan kepercayaan model untuk setiap kelas sampah.

### Cara Menjalankan Aplikasi Streamlit

1.  Pastikan Anda telah menginstal Streamlit dan dependensi Python lainnya:
    ```bash
    pip install -r requirements.txt
    ```
2.  Pastikan file model `best_model.keras` berada di direktori yang sama dengan skrip aplikasi Streamlit (`app.py` atau nama file `.py` Streamlit Anda).
3.  Buka terminal atau command prompt, navigasikan ke direktori proyek Anda.
4.  Jalankan aplikasi Streamlit dengan perintah:
    ```bash
    streamlit app.py
    ```
5.  Aplikasi akan terbuka di browser web default Anda.

### Tautan Aplikasi (Deployment)

Aplikasi Streamlit ini dapat diakses secara publik melalui tautan berikut:

[**Tautan Aplikasi Streamlit**](https://klasifikasi-sampah-ml.streamlit.app/)

## Kesimpulan

Model klasifikasi sampah ini menunjukkan kinerja yang sangat baik dengan akurasi tinggi pada data pengujian, berkat penggunaan Transfer Learning dengan Xception, augmentasi data yang cermat, dan strategi regulasi yang kuat untuk mencegah *overfitting*. Aplikasi Streamlit menyediakan antarmuka yang ramah pengguna untuk demonstrasi dan penggunaan praktis, berkontribusi pada pengelolaan limbah yang lebih baik.

-----
