# ATS Indonesian Election Domain

**Automatic Text Summarization (ATS)** berbasis **BART** untuk meringkas berita politik/pemilu berbahasa Indonesia secara *abstractive*, dengan fokus utama pada **pelatihan tokenizer khusus domain (domain-adaptive tokenizer)** guna meningkatkan kualitas ringkasan dibandingkan tokenizer bawaan BART.

Ide intinya sederhana: tokenizer BART asli dilatih pada korpus umum (mayoritas Bahasa Inggris), sehingga kurang efisien saat memproses teks berita pemilu Indonesia. Proyek ini menguji apakah melatih ulang tokenizer BPE pada korpus Bahasa Indonesia — termasuk korpus khusus domain pemilu — dapat meningkatkan performa model ringkasan.

## Daftar Isi
- [Gambaran Umum Eksperimen](#gambaran-umum-eksperimen)
- [Struktur Repository](#struktur-repository)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Cara Menjalankan](#cara-menjalankan)
- [Dependensi](#dependensi)
- [Catatan Git LFS](#catatan-git-lfs)

## Gambaran Umum Eksperimen

Proyek ini membandingkan **4 skenario tokenizer** yang digunakan untuk fine-tuning model `facebook/bart-base` pada tugas peringkasan teks:

| Skenario | Tokenizer | Dilatih dari |
|---|---|---|
| Baseline | Tokenizer bawaan BART | Korpus asli BART (multibahasa/Inggris) |
| Indomixedtufs4 | BPE kustom | Korpus umum Bahasa Indonesia (±165 ribu kalimat) |
| Detiknews-Pemilu | BPE kustom | Korpus berita pemilu dari Detik News (±165 ribu kalimat) |
| Mix | BPE kustom | Gabungan kedua korpus di atas (±330 ribu kalimat) |

Setiap skenario tokenizer digunakan untuk melatih ulang (fine-tune) model BART secara terpisah, lalu performanya dibandingkan menggunakan metrik ROUGE dan BERTScore.

## Struktur Repository

```
ats-indonesian-election-domain/
│
├── dataset/
│   ├── korpus-dataset/                     # Korpus mentah untuk melatih tokenizer BPE
│   │   ├── clean-indmixtufs4-165k-sent.txt      # Korpus umum Bahasa Indonesia
│   │   ├── clean-DetiknewsPemilu-165k-sent.txt  # Korpus domain berita pemilu
│   │   └── mix_dataset.txt                      # Gabungan kedua korpus
│   └── fine-tuning-dataset/
│       └── fine-tuning.csv                 # Dataset pasangan (artikel, ringkasan) — Git LFS
│
├── train-tokenizer/                        # Pelatihan tokenizer BPE per skenario
│   ├── 1. skenario-train-tokenizer-indomixedtufs4/
│   ├── 2. skenario-train-tokenizer-detiknews-pemilu/
│   ├── 3. skenario-train-tokenizer-mix/
│   └── output-bpe-model/                   # Tokenizer hasil pelatihan (tar.gz per skenario)
│
├── train-bart-model/                       # Fine-tuning BART per skenario tokenizer
│   ├── bartbase-barttokenizer.ipynb            # Baseline (tokenizer asli BART)
│   ├── finetune-indmixedtufs4-tokenizer.ipynb
│   ├── finetune-detiknews-tokenizer.ipynb
│   └── finetune-mix-tokenizer.ipynb
│
└── analisis/                               # Evaluasi & perbandingan hasil
    ├── perbandingan-tokenizer.ipynb            # Efisiensi tokenisasi (total token & fertility)
    ├── perbandingan-model.ipynb                # Perbandingan skor ROUGE & BERTScore
    └── generate-ringkasan.ipynb                # Demo inference: menghasilkan ringkasan
```

## Dataset

- **Korpus pelatihan tokenizer** (`dataset/korpus-dataset/`):
  - `clean-indmixtufs4-165k-sent.txt` — ±165 ribu kalimat, korpus umum Bahasa Indonesia (campuran berbagai sumber).
  - `clean-DetiknewsPemilu-165k-sent.txt` — ±165 ribu kalimat, korpus khusus berita pemilu dari Detik News.
  - `mix_dataset.txt` — gabungan kedua korpus di atas (±330 ribu kalimat).
- **Dataset fine-tuning** (`dataset/fine-tuning-dataset/fine-tuning.csv`): berisi kolom `text` (artikel berita) dan `summary` (ringkasan referensi), digunakan untuk fine-tuning BART maupun sebagai basis pengukuran efisiensi tokenizer. File ini dikelola lewat **Git LFS**.

## Metodologi

### 1. Pelatihan Tokenizer BPE Domain-Spesifik
Untuk tiap korpus (indomixedtufs4, detiknews-pemilu, mix), dilatih sebuah `ByteLevelBPETokenizer` dengan:
- `vocab_size = 50.265` (menyamai ukuran vocabulary asli BART, ada juga varian 10.000 dan 30.000 untuk perbandingan),
- `min_frequency = 2`,
- token spesial BART: `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`.

Setelah dilatih, indeks token `<mask>` disesuaikan agar berada di posisi terakhir vocabulary — meniru struktur asli tokenizer BART — sebelum tokenizer disimpan dan diunggah ke Kaggle Hub.

### 2. Fine-Tuning Model BART
Model dasar `facebook/bart-base` di-fine-tune terpisah untuk masing-masing dari 4 skenario tokenizer, dengan skema **5-fold cross-validation** dan konfigurasi berikut:

| Parameter | Nilai |
|---|---|
| Learning rate | 1e-5 |
| Batch size (train/eval) | 8 |
| Epoch | 10 |
| Precision | FP16 |
| Max input length | 1024 token |
| Max target length | 128 token |
| Trainer | `Seq2SeqTrainer` (Hugging Face) |

Metrik evaluasi yang dihitung tiap fold: **ROUGE-1, ROUGE-2, ROUGE-L**, dan **BERTScore** (precision, recall, F1).

### 3. Analisis & Evaluasi
- **`perbandingan-tokenizer.ipynb`** — mengukur efisiensi tokenisasi dengan menghitung total token yang dihasilkan tiap tokenizer pada dataset fine-tuning, serta **fertility metric** (rasio token terhadap jumlah kata asli — semakin rendah, semakin efisien).
- **`perbandingan-model.ipynb`** — memvisualisasikan perbandingan skor ROUGE dan BERTScore rata-rata pada data uji antar keempat skenario.
- **`generate-ringkasan.ipynb`** — memuat model hasil fine-tuning dari tiap skenario dan menghasilkan contoh ringkasan dari artikel berita pemilu, untuk perbandingan kualitatif.

## Hasil Eksperimen

### Efisiensi Tokenisasi (fertility metric)
Diukur pada ±3,8 juta kata (berdasarkan word tokenize bahasa Indonesia NLTK) dalam dataset fine-tuning:

| Tokenizer | Total Token | Fertility (token/kata) |
|---|---|---|
| BART original (baseline) | 8.886.348 | 2,34 |
| Indomixedtufs4 | 4.418.597 | 1,16 |
| Detiknews-Pemilu | 4.313.864 | 1,14 |
| Mix | 4.189.544 | **1,10** |

Semakin rendah nilai fertility, semakin efisien tokenizer memecah teks Bahasa Indonesia — tokenizer kustom terbukti jauh lebih hemat token dibanding tokenizer asli BART.

### Performa Model Ringkasan (skor rata-rata pada data uji)

| Skenario | ROUGE-1 | ROUGE-2 | BERTScore |
|---|---|---|---|
| Baseline (tokenizer BART asli) | 16,26 | 7,91 | 68,70 |
| Indomixedtufs4 | 24,56 | 12,41 | 71,23 |
| Detiknews-Pemilu | 25,36 | **12,95** | **71,39** |
| **Mix** | **25,40** | 12,85 | 71,32 |

**Kesimpulan utama**: mengganti tokenizer BART bawaan dengan tokenizer BPE yang dilatih khusus pada korpus Bahasa Indonesia memberikan peningkatan performa yang signifikan pada semua metrik. Skenario **Mix** (gabungan korpus umum + domain pemilu) memberikan skor ROUGE-1 tertinggi, sedangkan skenario **Detiknews-Pemilu** (murni domain-spesifik) unggul tipis pada ROUGE-2 dan BERTScore — menunjukkan bahwa adaptasi tokenizer ke domain target memberi manfaat nyata, dengan atau tanpa tambahan korpus umum.

## Cara Menjalankan

> Seluruh notebook awalnya dijalankan di lingkungan **Kaggle Notebook** (menggunakan GPU dan `kagglehub` untuk mengunduh/mengunggah dataset & model). Untuk menjalankan secara lokal, sesuaikan path `/kaggle/input/...` dengan lokasi file.

1. **Clone repository** (pastikan Git LFS sudah terpasang karena dataset fine-tuning dikelola lewat LFS):
   ```bash
   git lfs install
   git clone https://github.com/jawahirulfn/ats-indonesian-election-domain.git
   cd ats-indonesian-election-domain
   ```

2. **Latih tokenizer BPE** — jalankan salah satu notebook di `train-tokenizer/<skenario>/`, sesuaikan path korpus ke `dataset/korpus-dataset/`.

3. **Fine-tune model BART** — jalankan notebook terkait di `train-bart-model/`, arahkan path dataset ke `dataset/fine-tuning-dataset/fine-tuning.csv` dan path tokenizer ke hasil langkah sebelumnya (atau tokenizer di `train-tokenizer/output-bpe-model/`).

4. **Evaluasi & bandingkan hasil** — jalankan notebook di `analisis/` untuk melihat perbandingan efisiensi tokenizer, skor ROUGE/BERTScore antar model, serta contoh hasil ringkasan.

## Dependensi

Paket Python utama yang digunakan di seluruh notebook:

```
Cek file "requirements.txt
```

## Catatan Git LFS

File `dataset/fine-tuning-dataset/fine-tuning.csv` dilacak melalui **Git LFS** (`*.csv filter=lfs`). Jika setelah `git clone` isi file hanya berupa pointer teks (bukan data CSV sebenarnya), jalankan:
```bash
git lfs pull
```

---
