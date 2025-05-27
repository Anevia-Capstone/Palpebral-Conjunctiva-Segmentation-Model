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
Pastikan environment Python kamu sudah ter-install dependensi yang diperlukan, lalu jalankan script seperti biasa:

```bash
python Eye_and_Conjunctiva_Segmentation.ipynb
````

---
**Catatan:**
Jika kamu mengubah nama model atau lokasi folder, sesuaikan path-nya di script Python.

```
model.load_state_dict(torch.load("best.pt"))
```

```

Kalau kamu juga butuh versi Bahasa Indonesia atau auto-deteksi sistem operasi untuk otomatis download, aku bisa bantu tambahkan [versi lokalnya](f) atau [kode Python download otomatis](f).
```
