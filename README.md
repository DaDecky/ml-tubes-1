# Tugas Besar 1 IF3270 Pembelajaran Mesin

Repository ini berisi implementasi **Feedforward Neural Network (FFNN) from scratch** untuk Tugas Besar 1 IF3270 Pembelajaran Mesin. Implementasi utama berada di `src/ffnn` dan mencakup layer dense, fungsi aktivasi, fungsi loss, optimizer, regularisasi, normalisasi, penyimpanan model, serta notebook eksperimen untuk analisis pada dataset `global_student_placement_and_salary`.

## Struktur Singkat Repository

- `src/ffnn/`: implementasi modul FFNN
- `src/main.py`: contoh penggunaan model
- `src/Notebook_Testing.ipynb`: notebook eksperimen
- `src/datasetml_2026.csv`: dataset untuk pengujian
- `doc/Laporan.pdf`: laporan tugas besar

## Setup

Project ini menggunakan `uv` dan Python `>=3.11`. Disarankan menggunakan Python 3.11 agar tidak ada masalah kompatibilitas.

1. Buat dan sinkronkan environment:

```bash
uv sync
```

2. Aktifkan virtual environment sesuai sistem operasi:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## Menjalankan Program

Untuk menjalankan contoh program python:

```bash
python src/main.py
```

## Menjalankan Notebook Eksperimen

Notebook eksperimen berada di:

```bash
src/Notebook_Testing.ipynb
```

Jalankan Jupyter Notebook:

```bash
jupyter notebook src/Notebook_Testing.ipynb
```

## Pembagian Tugas

| Nama                | NIM      | Tugas                                                                                                                                         |
| ------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Muhammad Dicky Isra | 13523075 | Desain sistem, forward propagation, bonus initialization, laporan                                                                             |
| Guntara Hambali     | 13523114 | Backward propagation, optimizer, RMSNorm, laporan                                                                                             |
| Reza Ahmad Syarif   | 13523119 | Utility functions untuk backward propagation (loss, loss derivative, activation, activation derivative, L1, L2), notebook eksperimen, laporan |
