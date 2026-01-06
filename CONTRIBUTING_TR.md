# Visagen'e Katkıda Bulunma

<div align="center">

<img src="https://raw.githubusercontent.com/karasungur/visagen/main/assets/logo.png" alt="Visagen Logo" width="120"/>

**Visagen'e katkıda bulunmak istediğiniz için teşekkürler!**

*Her katkı, ne kadar küçük olursa olsun, bir fark yaratır.*

</div>

---

## 📋 İçindekiler

- [Davranış Kuralları](#-davranış-kuralları)
- [Başlarken](#-başlarken)
- [Geliştirme Ortamı](#-geliştirme-ortamı)
- [Değişiklik Yapma](#-değişiklik-yapma)
- [Commit Kuralları](#-commit-kuralları)
- [Pull Request Süreci](#-pull-request-süreci)
- [Kod Stili](#-kod-stili)
- [Test Yazma](#-test-yazma)
- [Dokümantasyon](#-dokümantasyon)
- [Issue Kuralları](#-issue-kuralları)
- [Tanınma](#-tanınma)

---

## 📜 Davranış Kuralları

Herkes için hoş karşılayıcı ve kapsayıcı bir ortam sağlamaya kararlıyız. Lütfen:

- **Saygılı olun** - Herkese saygı ve nezaketle davranın
- **Yapıcı olun** - Faydalı geri bildirim ve çözümlere odaklanın
- **Kapsayıcı olun** - Yeni gelenleri karşılayın ve başlamalarına yardımcı olun
- **Sabırlı olun** - Herkesin bir zamanlar başlangıç seviyesinde olduğunu unutmayın

---

## 🚀 Başlarken

### Ön Koşullar

- Python 3.10 veya üzeri
- Git
- (İsteğe bağlı) GPU hızlandırma için CUDA 11.8+

### Fork & Clone

1. GitHub'da repoyu **Fork**'layın
2. Fork'unuzu yerel olarak **Clone**'layın:

```bash
git clone https://github.com/KULLANICI_ADINIZ/visagen.git
cd visagen
```

3. **Upstream** remote ekleyin:

```bash
git remote add upstream https://github.com/karasungur/visagen.git
```

---

## 🔧 Geliştirme Ortamı

### Sanal Ortam Oluşturma

```bash
# Sanal ortam oluştur
python -m venv .venv

# Aktive et (Linux/Mac)
source .venv/bin/activate

# Aktive et (Windows)
.venv\Scripts\activate
```

### Bağımlılıkları Kurma

```bash
# Geliştirme bağımlılıkları ile kur
pip install -e ".[dev]"

# Tam geliştirme (tüm opsiyonel bağımlılıklar)
pip install -e ".[full,dev]"
```

### Kurulumu Doğrulama

```bash
# Her şeyin çalıştığını doğrulamak için testleri çalıştır
pytest visagen/tests/ -v --tb=short

# Kod stilini kontrol et
ruff check visagen/
```

---

## ✏️ Değişiklik Yapma

### 1. Branch Oluşturma

Değişiklikleriniz için her zaman yeni bir branch oluşturun:

```bash
# Önce main branch'i güncelle
git checkout main
git pull upstream main

# Özellik branch'i oluştur
git checkout -b feature/ozellik-adi

# Veya hata düzeltmeleri için
git checkout -b fix/hata-aciklamasi

# Veya dokümantasyon için
git checkout -b docs/degisiklik-aciklamasi
```

### 2. Branch İsimlendirme Kuralları

| Tip | Desen | Örnek |
|-----|-------|-------|
| Özellik | `feature/aciklama` | `feature/tensorrt-export-ekle` |
| Hata Düzeltme | `fix/aciklama` | `fix/decoder-bellek-sizintisi` |
| Dokümantasyon | `docs/aciklama` | `docs/api-referans-guncelle` |
| Refaktör | `refactor/aciklama` | `refactor/kayip-fonksiyonlari-sadele` |
| Test | `test/aciklama` | `test/entegrasyon-testleri-ekle` |

### 3. Değişikliklerinizi Yapın

- Temiz, okunabilir kod yazın
- Mevcut kod stilini takip edin
- Yeni işlevsellik için testler ekleyin
- Gerekirse dokümantasyonu güncelleyin

---

## 📝 Commit Kuralları

Temiz, anlamsal commit geçmişi için **Gitmoji + Conventional Commits** kullanıyoruz.

### Commit Mesajı Formatı

```
<emoji> <tip>(<kapsam>): <açıklama>

[isteğe bağlı gövde]

[isteğe bağlı altbilgi]
```

### Yaygın Commit Tipleri

| Emoji | Tip | Açıklama |
|:-----:|-----|----------|
| ✨ | `feat` | Yeni özellik |
| 🐛 | `fix` | Hata düzeltme |
| 📝 | `docs` | Dokümantasyon |
| 🎨 | `style` | Kod formatlama (mantık değişikliği yok) |
| ♻️ | `refactor` | Kod yeniden düzenleme |
| ⚡ | `perf` | Performans iyileştirme |
| ✅ | `test` | Test ekleme/güncelleme |
| 🔧 | `chore` | Bakım görevleri |
| 🏗️ | `build` | Build sistem değişiklikleri |
| 👷 | `ci` | CI/CD değişiklikleri |

### Örnekler

```bash
# Yeni özellik
git commit -m "✨ feat(export): TensorRT INT8 kuantizasyon desteği ekle"

# Hata düzeltme
git commit -m "🐛 fix(decoder): skip bağlantılarındaki bellek sızıntısını çöz"

# Dokümantasyon
git commit -m "📝 docs(readme): kurulum talimatlarını güncelle"

# Testler
git commit -m "✅ test(losses): gaze loss için birim testler ekle"
```

---

## 🔀 Pull Request Süreci

### Göndermeden Önce

1. **Upstream ile senkronize olun**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Testleri çalıştırın**:
   ```bash
   pytest visagen/tests/ -v
   ```

3. **Kod stilini kontrol edin**:
   ```bash
   ruff check visagen/
   ruff format visagen/
   ```

### PR Gönderme

1. Branch'inizi fork'unuza push edin:
   ```bash
   git push origin feature/ozellik-adi
   ```

2. GitHub'da Pull Request açın

3. PR şablonunu doldurun:
   - Değişikliklerin **açıklaması**
   - **İlgili issue'lar** (varsa)
   - **Yapılan testler**
   - **Ekran görüntüleri** (UI değişiklikleri için)

### PR İnceleme Süreci

- Tüm PR'lar en az bir inceleme gerektirir
- CI geçmelidir (testler, linting)
- İnceleyici geri bildirimlerini hızlıca ele alın
- PR'ları odaklı ve makul boyutta tutun

---

## 🎨 Kod Stili

`pyproject.toml`'da yapılandırılmış **Ruff** kullanıyoruz.

### Hızlı Komutlar

```bash
# Sorunları kontrol et
ruff check visagen/

# Sorunları otomatik düzelt
ruff check visagen/ --fix

# Kodu formatla
ruff format visagen/
```

### Stil Kuralları

- **Satır uzunluğu**: 88 karakter (Black varsayılanı)
- **Tırnaklar**: String'ler için çift tırnak
- **İmportlar**: isort kurallarıyla sıralı
- **Tip ipuçları**: Genel API'lar için gerekli
- **Docstring'ler**: Google stili

### Örnek Kod Stili

```python
"""Modül amacını açıklayan modül docstring'i."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class MyModule(nn.Module):
    """Sınıfın kısa açıklaması.

    Gerekirse daha uzun açıklama, sınıfın amacını
    ve kullanımını açıklar.

    Args:
        input_dim: Girdi özellik boyutu.
        hidden_dim: Gizli katman boyutu.

    Example:
        >>> module = MyModule(input_dim=64, hidden_dim=128)
        >>> output = module(torch.randn(1, 64))
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Girdi tensörünü işle.

        Args:
            x: (batch, input_dim) şeklinde girdi tensörü.

        Returns:
            (batch, hidden_dim) şeklinde çıktı tensörü.
        """
        return self.layer(x)
```

---

## 🧪 Test Yazma

### Testleri Çalıştırma

```bash
# Tüm testleri çalıştır
pytest visagen/tests/ -v

# Kapsam ile çalıştır
pytest visagen/tests/ --cov=visagen --cov-report=html

# Belirli test dosyasını çalıştır
pytest visagen/tests/test_forward_pass.py -v

# Desenle eşleşen testleri çalıştır
pytest visagen/tests/ -k "test_encoder" -v

# Sadece hızlı testleri çalıştır (yavaşları atla)
pytest visagen/tests/ -m "not slow" -v
```

### Test Yazma

- Testleri `visagen/tests/` içine yerleştirin
- Test dosyalarını `test_*.py` olarak adlandırın
- Test fonksiyonlarını `test_*` olarak adlandırın
- Açıklayıcı test isimleri kullanın

```python
"""Encoder modülü için testler."""

import pytest
import torch

from visagen.models.encoders.convnext import ConvNeXtEncoder


class TestConvNeXtEncoder:
    """ConvNeXtEncoder için test paketi."""

    def test_output_shape(self) -> None:
        """Encoder'ın doğru çıktı şekli ürettiğini test et."""
        encoder = ConvNeXtEncoder(in_channels=3)
        x = torch.randn(2, 3, 256, 256)

        features, latent = encoder(x)

        assert len(features) == 4
        assert latent.shape[0] == 2

    @pytest.mark.slow
    def test_large_input(self) -> None:
        """Büyük girdi ile encoder'ı test et (yavaş test)."""
        encoder = ConvNeXtEncoder(in_channels=3)
        x = torch.randn(1, 3, 1024, 1024)

        features, latent = encoder(x)

        assert latent is not None
```

---

## 📚 Dokümantasyon

### Docstring'ler

Tüm genel fonksiyonlar, sınıflar ve modüller docstring'lere sahip olmalıdır:

```python
def process_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: int = 512,
) -> np.ndarray:
    """Landmark'ları kullanarak görselden yüzü hizala ve kırp.

    Args:
        image: BGR numpy array olarak girdi görseli (H, W, 3).
        landmarks: (68, 2) şeklinde yüz landmark dizisi.
        target_size: Çıktı görsel boyutu (varsayılan: 512).

    Returns:
        BGR numpy array olarak hizalanmış yüz görseli (target_size, target_size, 3).

    Raises:
        ValueError: Landmark şekli geçersizse.

    Example:
        >>> face = process_face(image, landmarks, target_size=256)
        >>> assert face.shape == (256, 256, 3)
    """
```

### README Güncellemeleri

Yeni özellikler eklerken:
1. Uygunsa özellik listesini güncelleyin
2. Hızlı Başlangıç'a kullanım örnekleri ekleyin
3. Yeni komutlar ekliyorsanız CLI araçları tablosunu güncelleyin

---

## 🐛 Issue Kuralları

### Hata Raporlama

Lütfen şunları ekleyin:
- Sorunu açıklayan **net başlık**
- **Yeniden üretme adımları**
- **Beklenen davranış**
- **Gerçek davranış**
- **Ortam bilgisi** (OS, Python sürümü, GPU)
- **Hata mesajları** ve traceback'ler
- Uygunsa **ekran görüntüleri**

### Özellik İsteme

Lütfen şunları ekleyin:
- Özelliğin **net açıklaması**
- **Kullanım senaryosu** - neden gerekli?
- **Önerilen çözüm** (isteğe bağlı)
- **Değerlendirilen alternatifler** (isteğe bağlı)

---

## 🏆 Tanınma

Tüm katkıda bulunanlar README'mizde tanınır! Kod, dokümantasyon, hata raporları veya özellik istekleri olsun, katkılarınız değerlidir ve takdir edilir.

### Katkıda Bulunma Yolları

| Tip | Örnekler |
|-----|----------|
| 💻 Kod | Özellikler, hata düzeltmeleri, optimizasyonlar |
| 📝 Dokümantasyon | README, docstring'ler, eğitimler |
| 🐛 Hata Raporları | Sorunları bulma ve raporlama |
| 💡 Fikirler | Özellik istekleri, öneriler |
| 🧪 Test | Test yazma, PR'ları test etme |
| 🎨 Tasarım | UI/UX iyileştirmeleri |
| 🌍 Çeviri | Uluslararasılaştırma |

---

<div align="center">

## Sorularınız mı var?

Herhangi bir sorunuz varsa, bir [Issue](https://github.com/karasungur/visagen/issues) açmaktan çekinmeyin.

<br/>

**Keyifli Katkılar!** 🎉

<br/>

<sub>Visagen topluluğu tarafından ❤️ ile yapıldı</sub>

</div>
