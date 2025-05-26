# Palpebral-Conjunctiva-Segmentation-Model


## Conjunctiva Segmentation Workflow using YOLOv8

### 1. Dependency Installation

Pertama, dilakukan instalasi dependensi utama yaitu:

* `ultralytics` untuk menggunakan model YOLOv8 (termasuk task segmentasi),
* `roboflow` untuk mengambil dataset langsung dari Roboflow API.

### 2. Dataset Acquisition via Roboflow

Dataset segmentasi konjungtiva diunduh dari Roboflow menggunakan API key. Dataset ini telah diatur dalam format YOLOv8 segmentation dan terdiri dari tiga subset: `train`, `valid`, dan `test`.

### 3. Parsing Label Polygon YOLO

File label berupa polygon yang ter-normalisasi. Setiap baris label berisi class ID dan daftar koordinat polygon. Semua label dan informasi gambar diubah menjadi DataFrame untuk keperluan analisis lebih lanjut.

### 4. Training YOLOv8 Model (Segmentation)

Model YOLOv8 digunakan dalam mode segmentasi (`task='segment'`). Model yang digunakan dapat disesuaikan dengan kekuatan GPU (`yolov8s-seg.pt` untuk versi kecil, `yolov8m-seg.pt` untuk versi medium). Model dilatih menggunakan file konfigurasi `data.yaml` yang secara otomatis terbuat saat dataset dari Roboflow diunduh.

### 5. Evaluasi Awal Hasil Model

Model yang telah dilatih digunakan untuk melakukan inferensi pada 10 gambar dari subset `test`. Hasil segmentasi divisualisasikan dalam satu plot grid.

### 6. Ekstraksi dan Penyimpanan Area Konjungtiva

Hasil segmentasi dianalisis kembali untuk mengekstrak area dengan class `'conjunctiva'` saja:

* Mask biner dibuat dari polygon prediksi.
* ROI (region of interest) dipotong dari gambar asli berdasarkan bounding box mask.
* Hasil dipotong dan disimpan sebagai gambar baru yang hanya menampilkan area konjungtiva.

Langkah ini juga mencakup validasi untuk memastikan bahwa hanya area dengan mask valid yang diproses dan disimpan.

---

### Catatan Tambahan

* Semua proses berjalan di lingkungan Google Colab.
* Folder output otomatis dibuat untuk menyimpan hasil segmentasi.
* Proyek ini memanfaatkan model segmentasi YOLOv8 dan format polygon dari YOLO yang biasanya digunakan untuk instance segmentation.

---
