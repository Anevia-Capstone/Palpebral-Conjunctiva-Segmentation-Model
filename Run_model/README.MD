# Petunjuk menjalankan model
## Download Pretrained Model (.pt)

Untuk menjalankan model segmentasi ini, silakan ikuti langkah-langkah berikut:

### 1. Buka Halaman Releases
Kunjungi halaman **GitHub Releases** di repositori ini:

[Releases > ModelSegmentasi](https://github.com/Anevia-Capstone/Palpebral-Conjunctiva-Segmentation-Model/releases/tag/ModelSegmentasi)

### 2. Unduh Model
Di halaman tersebut, unduh file berikut:

```

best.pt

```

### 3. Simpan File Model
Letakkan file `best.pt` yang sudah diunduh ke dalam direktori proyek ini, **di folder yang sama dengan `main.py`**.

Contoh struktur folder:

```

Palpebral-Conjunctiva-Segmentation-Model/Run_model
├── best.pt      <- taruh di sini
├── main.py
├── requirements.txt
├── README.md

````

### 4. Jalankan Model
Pastikan environment Python kamu sudah ter-install dependensi yang diperlukan:

```bash
pip install -r requirements.txt
````

---
**Catatan:**
Jalankan kodenya

```
python main.py
```
