<div align="center">

<!-- Animated Header Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=120&section=header" width="100%"/>

<!-- Logo -->
<img src="assets/logo.png" alt="Visagen Logo" width="180"/>

<!-- Animated Title -->
<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=667EEA&center=true&vCenter=true&width=500&lines=Visagen;Next-Gen+Face+Swapping;PyTorch+Lightning+Powered" alt="Typing SVG" />
</h1>

<p><em>Modern Face Swapping Framework with ConvNeXt, CBAM & Lightning</em></p>

<!-- Language Selector -->
<p>
  <a href="README.md">🇺🇸 English</a> |
  <a href="README_TR.md">🇹🇷 Türkçe</a>
</p>

<!-- Badges -->
<p>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Lightning-2.0%2B-792EE5?style=for-the-badge&logo=lightning&logoColor=white" alt="Lightning"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/lisans-MIT-00C853?style=for-the-badge" alt="Lisans: MIT"/></a>
</p>

<!-- Hızlı Navigasyon -->
<p>
  <a href="#-özellikler">✨ Özellikler</a> •
  <a href="#-kurulum">📦 Kurulum</a> •
  <a href="#-hızlı-başlangıç">🚀 Hızlı Başlangıç</a> •
  <a href="#%EF%B8%8F-cli-araçları">🛠️ CLI Araçları</a> •
  <a href="#%EF%B8%8F-mimari">🏗️ Mimari</a> •
  <a href="#-katkıda-bulunma">🤝 Katkıda Bulunma</a>
</p>

<br/>

</div>

---

## 📖 Genel Bakış

**Visagen**, modern derin öğrenme pratikleri ile sıfırdan inşa edilmiş yeni nesil bir yüz değiştirme framework'üdür. DeepFaceLab'dan ilham alınarak, Visagen tüm pipeline'ı **PyTorch Lightning** kullanarak yeniden tasarlar ve daha temiz kod, daha iyi performans ve kolay genişletilebilirlik sunar.

```
┌───────────────────────────────────────────────────────────────┐
│                       VISAGEN PIPELINE                        │
├───────────────────────────────────────────────────────────────┤
│  Çıkart    ──►   Eğit    ──►  Değiştir  ──►   Son İşlem       │
│     │              │            │              │              │
│     ▼              ▼            ▼              ▼              │
│ InsightFace    DFLModule      CBAM       Renk Transferi       │
│ SegFormer     Lightning    Attention      Harmanlama          │
└───────────────────────────────────────────────────────────────┘
```

---

## ✨ Özellikler

<table>
<tr>
<td width="50%">

### 🧠 Modern Mimari
- GRN katmanlı **ConvNeXt V2** encoder
- **CBAM** attention (Kanal & Uzamsal)
- Yumuşak gradyanlar için **Swish** aktivasyon
- Detay koruma için skip bağlantıları

</td>
<td width="50%">

### 🎯 Gelişmiş Eğitim
- Çoklu kayıp: DSSIM, L1, LPIPS, ID, GAN
- Karışık hassasiyet (FP16/BF16)
- Gradyan kırpma & LR zamanlama
- Göz/Ağız & Bakış tutarlılık kaybı

</td>
</tr>
<tr>
<td width="50%">

### 🎨 Son İşleme
- 6 renk transferi algoritması (RCT, LCT, SOT...)
- Nöral renk transferi (VGG tabanlı)
- Laplacian, Poisson, Feather harmanlama
- GFPGAN & GPEN yüz restorasyon

</td>
<td width="50%">

### ⚡ Üretime Hazır
- ONNX & TensorRT dışa aktarım
- NVENC donanım kodlama
- 30+ FPS çıkarım hızı
- 12 CLI aracı

</td>
</tr>
</table>

---

## 📦 Kurulum

### Gereksinimler
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU hızlandırma için)

### Temel Kurulum

```bash
# Repoyu klonla
git clone https://github.com/karasungur/visagen.git
cd visagen

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Temel paketi kur
pip install -e .
```

### Tam Kurulum (Tüm Özellikler)

```bash
pip install -e ".[full]"
```

<details>
<summary><b>📋 Opsiyonel Bağımlılıklar</b></summary>

```bash
# Görü (InsightFace, SegFormer)
pip install -e ".[vision]"

# Eğitim (LPIPS)
pip install -e ".[training]"

# Hiperparametre Ayarlama (Optuna)
pip install -e ".[tuning]"

# Web Arayüzü (Gradio)
pip install -e ".[gui]"

# Son İşleme (Renk Transferi, Harmanlama)
pip install -e ".[postprocess]"

# Video Birleştirici (FFmpeg)
pip install -e ".[merger]"

# Model Dışa Aktarım (ONNX, TensorRT)
pip install -e ".[export]"

# Yüz Restorasyon (GFPGAN)
pip install -e ".[restore]"

# GPU Veri Yükleme (NVIDIA DALI)
pip install -e ".[dali]"
```

</details>

---

## 🚀 Hızlı Başlangıç

<details open>
<summary><b>📥 Adım 1: Yüz Çıkarma</b></summary>

```bash
# Videodan çıkar
visagen-extract \
    --input video.mp4 \
    --output-dir ./workspace/data_src/aligned \
    --face-size 512

# Görsellerden çıkar
visagen-extract \
    --input ./photos/ \
    --output-dir ./workspace/data_dst/aligned \
    --face-size 512
```

</details>

<details open>
<summary><b>🏋️ Adım 2: Model Eğitimi</b></summary>

```bash
visagen-train \
    --src-dir ./workspace/data_src/aligned \
    --dst-dir ./workspace/data_dst/aligned \
    --output-dir ./workspace/model \
    --epochs 500 \
    --batch-size 8 \
    --resolution 512
```

</details>

<details>
<summary><b>🔧 Adım 3: Hiperparametre Ayarlama (İsteğe Bağlı)</b></summary>

```bash
visagen-tune \
    --src-dir ./workspace/data_src/aligned \
    --dst-dir ./workspace/data_dst/aligned \
    --output-dir ./workspace/optuna \
    --n-trials 20 \
    --epochs-per-trial 50
```

</details>

<details>
<summary><b>🎬 Adım 4: Yüzleri Videoya Birleştir</b></summary>

```bash
# Eğitilmiş model ile temel birleştirme
visagen-merge input.mp4 output.mp4 -c ./workspace/model/model.ckpt

# Yüz restorasyon ve donanım kodlama ile
visagen-merge input.mp4 output.mp4 -c model.ckpt \
    --restore-face --restore-strength 0.7 \
    --codec h264_nvenc --color-transfer rct
```

</details>

<details>
<summary><b>📦 Adım 5: Üretim için Model Dışa Aktarımı</b></summary>

```bash
# ONNX'e aktar
visagen-export model.ckpt -o model.onnx --validate

# TensorRT'ye aktar (FP16)
visagen-export model.onnx -o model.engine --format tensorrt --precision fp16
```

</details>

<details>
<summary><b>🌐 Adım 6: Web Arayüzünü Başlat</b></summary>

```bash
visagen-gui --port 7860
```

</details>

---

## 🏗️ Mimari

### Model Mimarisi

```
Girdi (512x512x3)
       │
       ▼
┌──────────────┐
│   ConvNeXt   │  ← Encoder (önceden eğitilmiş)
│   Encoder    │
└──────────────┘
       │
       ▼
┌──────────────┐
│    CBAM      │  ← Kanal & Uzamsal Attention
│  Attention   │
└──────────────┘
       │
       ▼
┌──────────────┐
│    Swish     │  ← Skip bağlantılı Decoder
│   Decoder    │
└──────────────┘
       │
       ▼
Çıktı (512x512x3)
```

---

## 🛠️ CLI Araçları

| Komut | Açıklama |
|:------|:---------|
| 📥 `visagen-extract` | Görsel/videodan yüz çıkar ve hizala |
| 🏋️ `visagen-train` | Yüz değiştirme modeli eğit |
| 🎯 `visagen-pretrain` | FFHQ/CelebA üzerinde encoder ön-eğitimi |
| 🔧 `visagen-tune` | Hiperparametre optimizasyonu (Optuna) |
| 🎬 `visagen-merge` | NVENC kodlama ile yüz değiştirme birleştir |
| 📦 `visagen-export` | ONNX/TensorRT'ye aktar |
| 📊 `visagen-sort` | Veri seti sırala (14 yöntem) |
| 🌐 `visagen-gui` | Gradio web arayüzü başlat |
| 🎞️ `visagen-video` | Video kare çıkarma/oluşturma |
| ✨ `visagen-enhance` | Toplu yüz iyileştirme (GFPGAN/GPEN) |
| 📐 `visagen-resize` | Metadata ile faceset boyutlandır |
| ⚡ `visagen-benchmark` | Performans karşılaştırmaları |

---

<details>
<summary><b>📁 Proje Yapısı</b></summary>

```
visagen/
├── 📂 data/               # Veri yükleme & augmentasyon
│   ├── dataset.py         # FaceDataset
│   ├── datamodule.py      # FaceDataModule
│   ├── dali_pipeline.py   # NVIDIA DALI GPU pipeline
│   └── augmentations.py
├── 📂 models/             # Sinir ağı mimarileri
│   ├── encoder.py         # ConvNeXt encoder
│   ├── decoder.py         # Swish decoder
│   ├── attention.py       # CBAM attention
│   └── discriminator.py
├── 📂 training/           # Eğitim mantığı
│   ├── dfl_module.py      # PyTorch Lightning modülü
│   ├── pretrain_module.py # Ön-eğitim modülü
│   └── losses.py          # Kayıp fonksiyonları
├── 📂 merger/             # Video işleme pipeline
│   ├── video_io.py        # NVENC ile FFmpeg video I/O
│   ├── frame_processor.py # Tek-kare işleme
│   ├── batch_processor.py # Paralel işleme
│   └── merger.py          # Üst düzey orkestrasyon
├── 📂 postprocess/        # Son işleme
│   ├── color_transfer.py  # RCT, LCT, SOT, MKL, IDT algoritmaları
│   ├── neural_color.py    # VGG tabanlı nöral renk transferi
│   ├── blending.py        # Laplacian, Poisson, Feather
│   ├── restore.py         # GFPGAN yüz restorasyon
│   └── gpen.py            # GPEN yüz restorasyon
├── 📂 export/             # Model dışa aktarım
│   ├── onnx_exporter.py   # ONNX dışa aktarım
│   ├── tensorrt_builder.py# TensorRT motor oluşturucu
│   └── validation.py      # Dışa aktarım doğrulama
├── 📂 sorting/            # Veri seti sıralama
│   └── sorter.py          # 14 sıralama yöntemi
├── 📂 tuning/             # Hiperparametre optimizasyonu
│   └── optuna_tuner.py
├── 📂 tools/              # CLI araçları
│   ├── extract_v2.py      # Yüz çıkarma
│   ├── train.py           # Eğitim
│   ├── pretrain.py        # Ön-eğitim
│   ├── merge.py           # Video birleştirme
│   ├── export.py          # Model dışa aktarım
│   ├── sorter.py          # Veri seti sıralama
│   ├── tune.py            # HPO
│   ├── video_ed.py        # Video kare araçları
│   ├── faceset_enhancer.py# Toplu yüz iyileştirme
│   ├── faceset_resizer.py # Faceset boyutlandırma
│   ├── benchmark.py       # Performans karşılaştırmaları
│   └── gradio_app.py      # Web UI (10 sekme)
├── 📂 vision/             # Bilgisayarlı görü
│   ├── detector.py        # InsightFace SCRFD algılama
│   ├── aligner.py         # Yüz hizalama (Umeyama)
│   ├── segmenter.py       # SegFormer segmentasyon
│   ├── dflimg.py          # DFL görsel metadata
│   └── mask_export.py     # LabelMe/COCO dışa aktarım
└── 📂 tests/              # Birim testleri (636+)
```

</details>

---

## 📊 Performans

<table>
<tr>
<td align="center">
<h3>🚄 50 img/s</h3>
<sub>Eğitim Hızı (RTX 3090)</sub>
</td>
<td align="center">
<h3>💾 8 GB</h3>
<sub>VRAM (512×512, batch=8)</sub>
</td>
<td align="center">
<h3>⚡ 30 FPS</h3>
<sub>Çıkarım Hızı</sub>
</td>
<td align="center">
<h3>✅ 636+</h3>
<sub>Birim Test</sub>
</td>
</tr>
</table>

---

## 👥 Katkıda Bulunanlar

<table>
<tr>
<td align="center">
  <a href="https://github.com/karasungur">
    <img src="https://github.com/karasungur.png" width="60px;" alt="Mustafa Karasungur"/><br />
    <sub><b>Mustafa Karasungur</b></sub>
  </a><br />
  <sub>🏗️ Proje Lideri</sub>
</td>
</tr>
</table>

<p align="center">
  <i>Katkılar her zaman hoş karşılanır! Aşağıdaki bölüme bakın.</i>
</p>

---

## 🤝 Katkıda Bulunma

Katkılarınızı seviyoruz! İster hata düzeltme, ister dokümantasyon iyileştirme, ister yeni özellik önerisi olsun, yardımınız değerlidir.

<details>
<summary><b>📋 Katkıda Bulunanlar için Hızlı Başlangıç</b></summary>

```bash
# Repoyu forkla ve klonla
git clone https://github.com/KULLANICI_ADINIZ/visagen.git
cd visagen

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Geliştirme kurulumu
pip install -e ".[dev]"

# Yeni branch oluştur
git checkout -b feature/ozellik-adi
```

</details>

<details>
<summary><b>🧪 Testleri Çalıştırma</b></summary>

```bash
# Tüm testleri çalıştır
pytest visagen/tests/ -v

# Kapsam ile çalıştır
pytest visagen/tests/ --cov=visagen --cov-report=html

# Belirli test dosyasını çalıştır
pytest visagen/tests/test_forward_pass.py -v
```

</details>

<details>
<summary><b>🎨 Kod Stili</b></summary>

Linting ve formatlama için **Ruff** kullanıyoruz:

```bash
# Kod stilini kontrol et
ruff check visagen/

# Otomatik formatla
ruff format visagen/

# Otomatik düzeltilebilir sorunları düzelt
ruff check visagen/ --fix
```

</details>

Detaylı kurallar için [**Katkıda Bulunma Rehberi**](CONTRIBUTING_TR.md)'ne bakın.

---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkürler

- [DeepFaceLab](https://github.com/iperov/DeepFaceLab) - Orijinal ilham kaynağı
- [PyTorch Lightning](https://lightning.ai/) - Eğitim framework'ü
- [InsightFace](https://github.com/deepinsight/insightface) - Yüz algılama
- [Optuna](https://optuna.org/) - Hiperparametre optimizasyonu

---

<div align="center">

**[Mustafa Karasungur](https://github.com/karasungur) tarafından ❤️ ile yapıldı**

<br/>

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=100&section=footer" width="100%"/>

</div>
