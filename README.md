# Proyek: Segmentasi Konjungtiva Palpebral Menggunakan UNet + ResNet50

---

## Project Overview (Ulasan Proyek)

### **Latar Belakang**

Segmentasi konjungtiva palpebral adalah proses penting dalam bidang medis, khususnya di opthalmologi. Konjungtiva palpebral merupakan bagian dari lapisan transparan yang melapisi kelopak mata dan berperan penting dalam diagnosis kondisi seperti anemia, infeksi, atau defisiensi nutrisi. Dengan kemajuan teknologi komputer visi dan deep learning, segmentasi otomatis menjadi solusi efektif untuk menggantikan metode manual yang memakan waktu dan rentan terhadap kesalahan manusia.

Penelitian oleh Prakash et al. (2022) mengembangkan sistem berbasis deep learning untuk mendeteksi anemia dari citra konjungtiva palpebral. Mereka mengumpulkan dataset gambar mata menggunakan kamera ponsel dalam pencahayaan alami, lalu menerapkan model CNN (Convolutional Neural Network) untuk klasifikasi tingkat hemoglobin rendah. Hasil menunjukkan bahwa model CNN mampu mendeteksi anemia dengan akurasi yang cukup tinggi tanpa perlu prosedur invasif seperti pengambilan darah. Penelitian ini menekankan potensi teknologi mobile dan deep learning sebagai alat skrining awal anemia, terutama di daerah dengan akses medis terbatas.

Studi lain dilakukan oleh Islam et al. (2021), yang memanfaatkan teknik deep convolutional neural networks untuk klasifikasi anemia berdasarkan citra konjungtiva. Dengan dataset gambar mata yang telah dianotasi oleh ahli medis, mereka membandingkan berbagai arsitektur CNN termasuk ResNet dan VGG, dan menemukan bahwa ResNet-50 memberikan performa terbaik. Sistem ini menunjukkan kemampuan untuk mendeteksi anemia dengan sensitivitas yang baik, menandakan bahwa pendekatan ini berpotensi digunakan sebagai alat diagnostik awal yang cepat dan murah. Penelitian ini juga memperhatikan pentingnya segmentasi area konjungtiva untuk meningkatkan akurasi klasifikasi.


Proyek ini bertujuan untuk membuat model segmentasi otomatis menggunakan arsitektur U-Net dengan backbone ResNet50 untuk mendeteksi area konjungtiva palpebral pada gambar mata manusia. Model ini dapat digunakan sebagai dasar untuk sistem pendukung keputusan medis atau aplikasi mobile screening berbasis AI.


### **Mengapa Proyek Ini Penting?**

1. **Akurasi Diagnostik**: Segmentasi area konjungtiva membantu dokter dalam mengevaluasi warna dan kondisi selaput lendir, yang bisa menjadi indikator penyakit tertentu.
2. **Efisiensi Waktu**: Automatisasi akan mengurangi beban kerja tenaga medis dan meningkatkan throughput pemeriksaan.
3. **Akses Kesehatan yang Lebih Luas**: Dengan penggunaan smartphone untuk pengambilan gambar, sistem ini bisa diterapkan di daerah dengan akses kesehatan terbatas.

### **Referensi Penelitian Terkait**
- Dataset berasal dari penelitian oleh [Mohammad Marufur Rahman](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=wN0nqFwAAAAJ&citation_for_view=wN0nqFwAAAAJ:Y0pCki6q_DkC), yang menjelaskan metodologi pengumpulan dan labeling data.
- Dataset tersedia di Mendeley: [Conjunctiva Segmentation Dataset - Mendeley Data](https://data.mendeley.com/datasets/yxwjgcndg2/1).

---

## Business Understanding

### **Problem Statement (Pernyataan Masalah)**

> “Bagaimana cara melakukan segmentasi otomatis area konjungtiva palpebral pada gambar mata manusia dengan akurasi tinggi agar dapat digunakan untuk analisis lebih lanjut dalam sistem pendukung keputusan medis?”

### **Tujuan (Goals)**

1. Membuat model segmentasi berbasis deep learning untuk mendeteksi area konjungtiva palpebral secara akurat.
2. Memvalidasi performa model menggunakan metrik IoU dan Dice Score.
3. Menyediakan visualisasi hasil segmentasi dan overlay masker.
4. Menyimpan model terbaik untuk inferensi baru dan implementasi di lingkungan nyata.

### **Solution Approach**

Dalam konteks proyek ini, tidak ada dua pendekatan rekomendasi karena proyek bersifat *image segmentation*. Namun jika Anda ingin menyambung ke sistem rekomendasi misalnya untuk pemilihan model terbaik, pendekatan yang mungkin adalah:

#### 1. Content-Based Filtering
Model dipilih berdasarkan karakteristik input gambar (seperti resolusi, cahaya, dll.) dan output yang dihasilkan (IoU, Dice, dll.).

#### 2. Collaborative Filtering
Rekomendasi model dilakukan berdasarkan performa relatif model sebelumnya terhadap dataset serupa dari repository lain (misalnya PyPI, GitHub, HuggingFace).

Namun dalam kasus ini, fokus utama adalah pembuatan dan evaluasi model segmentasi.

---

## Data Understanding

### **Sumber Data**
[Dataset Tersedia di Mendeley](https://data.mendeley.com/datasets/yxwjgcndg2/1)

### **Deskripsi Dataset**
- **Jumlah Gambar:** 547 gambar mata manusia.
- **Resolusi Rata-Rata:** ~3000x4000 piksel.
- **Device Pengambilan Gambar:** OnePlus 9R (48 MP) dan OnePlus 9Pro (48 MP).
- **Kondisi Pencahayaan:** Ruangan dengan pencahayaan alami, tanpa lampu tambahan.
- **Folder Struktur:**
  - `Images`: Berisi 547 gambar asli.
  - `Masks Annotator 1`: Label segmen dari anotator pertama.
  - `Masks Annotator 2`: Label segmen dari anotator kedua.
- **Inter-Annotator Agreement (IAA):** Rata-rata 99.9% overlap antara dua anotator, menunjukkan kualitas label sangat tinggi.

### **Variabel / Fitur**
| Nama Folder | Deskripsi |
|-------------|-----------|
| Images | Gambar asli dari mata manusia yang menampilkan konjungtiva palpebral |
| Masks Annotator 1 & 2 | Masker biner (hitam-putih) yang menunjukkan lokasi konjungtiva hasil anotasi |

### **Visualisasi dan EDA Awal**
- Gambar-gambar memiliki variasi posisi mata, warna kulit, dan pencahayaan.
- Masker memiliki ukuran yang sama dengan gambar asli.
- Semua gambar disimpan dalam format `.png`.

---

## Data Preparation

### **Teknik yang Digunakan**
1. **Resize Gambar**:
   - Ukuran gambar asli terlalu besar (3000x4000). Untuk efisiensi pelatihan, semua gambar diubah menjadi ukuran `512x512` pixel.

2. **Split Dataset**:
   - Data dibagi menjadi set pelatihan (`train`) dan validasi (`val`) dengan rasio 80:20 (437 train, 110 val).

3. **Augmentasi Data (Train Set)**:
   - Flip horizontal/vertikal
   - Rotasi acak
   - Perubahan kecerahan/kontras
   - Distorsi geometri
   - Noise tambahan

4. **Normalisasi Input**:
   - Normalisasi menggunakan mean `[0.485, 0.456, 0.406]` dan std `[0.229, 0.224, 0.225]` sesuai ImageNet.

5. **Label Biner**:
   - Masker dikonversi ke biner (`float32`, 0 = bukan konjungtiva, 1 = konjungtiva).

### **Alasan Data Preparation**
- Resize: Agar model dapat dilatih lebih cepat tanpa kehilangan informasi signifikan.
- Augmentasi: Meningkatkan variasi data dan mengurangi overfitting.
- Normalisasi: Memastikan distribusi nilai input sesuai dengan model pre-trained (ImageNet).

---

## Modeling and Result

### **Arsitektur Model**
- **Model Utama**: U-Net
- **Encoder Backbone**: ResNet50 (pretrained pada ImageNet)
- **Input Channel**: 3 (RGB)
- **Output Class**: 1 (biner: konjungtiva atau bukan)

### **Fungsi Loss**
```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, pred, target):
        return self.alpha * self.dice_loss(pred, target) + self.beta * self.focal_loss(pred, target)
```

### **Metrik Evaluasi**
- **IoU (Intersection over Union)**
- **Dice Score**
- **Binary Accuracy**

### **Hyperparameter**
- Optimizer: AdamW
- Learning Rate: 1e-4 → Cosine Annealing hingga 1e-6
- Batch Size: 8
- Epoch: 20
- Device: CUDA (GPU)

### **Hasil Pelatihan**
- **Best Validation IoU**: 0.7453
- **Best Validation Dice Score**: 0.8528
- **Final Train Loss**: 0.3469
- **Final Val Loss**: 0.3472

### **Visualisasi Hasil**
Setelah Training, model menghasilkan:
![image](https://github.com/user-attachments/assets/82d2f651-9406-432a-86be-7366523ed8ee)

### **Hasil Inferensi**
Setelah inferensi, model menghasilkan:
![image](https://github.com/user-attachments/assets/a1440896-7a3e-4a65-a9a4-c67aff34770e)

![image](https://github.com/user-attachments/assets/e171ae32-2b2d-4cf6-bd04-935ebd376a8c)


---

## Evaluation

### **Metrik Evaluasi yang Digunakan**
| Metrik | Formula | Tujuan |
|--------|---------|--------|
| **IoU (Intersection over Union)** | $ \text{IoU} = \frac{\text{Intersection}}{\text{Union}} $ | Mengukur overlap antara prediksi dan ground truth |
| **Dice Score** | $ \text{Dice} = \frac{2 \cdot \text{Intersection}}{\text{Total Area}} $ | Mirip dengan IoU namun lebih toleran terhadap ketidakseimbangan kelas |

### **Hasil Evaluasi Model**
- Model berhasil mencapai IoU > 0.7 dan Dice > 0.85 pada data validasi, menunjukkan bahwa model cukup baik dalam menemukan area konjungtiva.
- Tidak ada overfitting signifikan karena gap antara train dan val loss tetap kecil.
- Performa stabil meskipun learning rate diturunkan secara bertahap.

---

## Kesimpulan

Model U-Net dengan encoder ResNet50 telah berhasil dilatih untuk melakukan segmentasi konjungtiva palpebral dengan akurasi yang tinggi. Dengan IoU hingga 0.7453 dan Dice Score 0.8528, model siap digunakan untuk inferensi pada gambar baru. Teknik augmentasi dan fungsi loss gabungan (CombinedLoss) memberikan kontribusi besar dalam meningkatkan generalisasi model.

---
## Referensi 

Prakash, P., Kaur, A., & Saini, H. (2022). Deep Learning-Based Non-Invasive Anemia Detection Using Palpebral Conjunctiva Images. Computational Intelligence and Neuroscience, 2022, Article ID 8653021. https://doi.org/10.1155/2022/8653021

Islam, M. T., Wahid, K. A., & Yafi, C. (2021). Automated Detection of Anemia Using Deep Learning on Images of the Conjunctiva. Healthcare Technology Letters, 8(2), 29–35. https://doi.org/10.1049/htl2.12005
