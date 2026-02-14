"""Türkçe çeviriler."""

TRANSLATIONS = {
    # Ortak
    "common": {
        "start": "Başlat",
        "stop": "Durdur",
        "save": "Kaydet",
        "load": "Yükle",
        "cancel": "İptal",
        "apply": "Uygula",
        "refresh": "Yenile",
        "export": "Dışa Aktar",
        "browse": "Gözat",
    },
    # Uygulama
    "app": {
        "title": "Visagen - Yüz Değiştirme Uygulaması",
        "subtitle": "PyTorch Lightning ile modern YÜz Değiştirme uygulaması",
        "footer": "Visagen v2.0.0-alpha | PyTorch Lightning ile güçlendirildi.",
    },
    # İş Akışı
    "workflow": {
        "title": "İş Akışı Adımları",
        "steps": {
            "extract": "Çıkar",
            "sort": "Sırala",
            "train": "Eğit",
            "merge": "Birleştir",
            "export": "Dışa Aktar",
        },
        "descriptions": {
            "extract": "Video veya resimlerden yüzleri çıkar",
            "sort": "Çıkarılan yüzleri filtrele ve düzenle",
            "train": "Yüz değiştirme modelini eğit",
            "merge": "Eğitilmiş modeli videolara uygula",
            "export": "İşlenmiş videoları dışa aktar",
        },
    },
    # Hatalar
    "errors": {
        "path_required": "Yol gerekli",
        "path_not_found": "Yol bulunamadı",
        "not_a_directory": "Yol bir dizin değil",
        "invalid_file_type": "Geçersiz dosya türü. Beklenen: {types}",
        "no_model_loaded": "Model yüklenmedi. Lütfen önce bir checkpoint yükleyin.",
        "process_failed": "İşlem {code} çıkış koduyla başarısız oldu",
        "missing_images": "Gerekli görüntüler eksik",
        # Yeni hata mesajları
        "no_output_dir": "Çıktı dizini belirtilmedi",
        "source_image_required": "Lütfen bir kaynak görüntü sağlayın",
    },
    # Durum mesajları
    "status": {
        "ready": "Hazır",
        "loading": "Yükleniyor...",
        "processing": "İşleniyor...",
        "completed": "Tamamlandı",
        "failed": "Başarısız",
        "model_loaded": "Model yüklendi: {name}",
        "model_unloaded": "Model kaldırıldı",
        "no_model": "Model yüklenmedi",
        "files_found": "{count} dosya bulundu",
        "stopped": "İşlem kullanıcı tarafından durduruldu",
        # Yeni durum mesajları
        "no_training": "Devam eden eğitim yok",
        "no_merge": "Devam eden birleştirme yok",
        "no_sorting": "Devam eden sıralama yok",
        "no_extraction": "Devam eden çıkarma yok",
        "no_export": "Devam eden dışa aktarma yok",
        "preview_available": "Önizleme mevcut",
        "extraction_completed": "Çıkarma başarıyla tamamlandı!",
        "extraction_stopped": "Çıkarma durduruldu",
    },
    # Eğitim sekmesi
    "training": {
        "title": "Model Eğitimi",
        "src_dir": {
            "label": "Kaynak Dizin",
            "placeholder": "./workspace/data_src/aligned",
            "info": "Kaynak yüz görüntülerini içeren dizin",
        },
        "dst_dir": {
            "label": "Hedef Dizin",
            "placeholder": "./workspace/data_dst/aligned",
            "info": "Hedef yüz görüntülerini içeren dizin",
        },
        "output_dir": {
            "label": "Çıktı Dizini",
            "info": "Checkpoint ve loglar için dizin",
        },
        "batch_size": {
            "label": "Batch Boyutu",
        },
        "max_epochs": {
            "label": "Maksimum Epoch",
        },
        "learning_rate": {
            "label": "Öğrenme Oranı",
        },
        "start": "Eğitimi Başlat",
        "stop": "Eğitimi Durdur",
        "log": {
            "label": "Eğitim Logu",
        },
        "preview": {
            "title": "Eğitim Önizleme",
            "status": {
                "label": "Önizleme Durumu",
            },
            "image": {
                "label": "Önizleme Izgarası",
            },
        },
        "dssim_weight": {
            "label": "DSSIM Ağırlığı",
        },
        "l1_weight": {
            "label": "L1 Ağırlığı",
        },
        "lpips_weight": {
            "label": "LPIPS Ağırlığı",
            "info": "lpips paketi gerektirir",
        },
        "gan_power": {
            "label": "GAN Gücü",
            "info": "0 = devre dışı, > 0 = çekişmeli eğitim",
        },
        "eyes_mouth_weight": {
            "label": "Göz/Ağız Ağırlığı",
            "info": "Göz ve ağız bölgeleri için öncelik (0-300)",
        },
        "gaze_weight": {
            "label": "Bakış Ağırlığı",
            "info": "Bakış tutarlılık kaybı (landmark gerektirir)",
        },
        "face_style_weight": {
            "label": "Yüz Stili Ağırlığı",
            "info": "Maske içinde hedef yüz rengini öğren (0-100)",
        },
        "bg_style_weight": {
            "label": "Arka Plan Stili Ağırlığı",
            "info": "Maske dışında hedef arka planı öğren (0-100)",
        },
        "true_face_power": {
            "label": "Gerçek Yüz Gücü",
            "info": "Kimlik ayrıştırıcı (sadece df mimarisi, 0-1)",
        },
        "id_weight": {
            "label": "Kimlik Ağırlığı",
            "info": "ArcFace ile kimlik koruma kaybı (0-1, insightface gerektirir)",
        },
        "temporal_power": {
            "label": "Zamansal Güç",
            "info": "Zamansal ayrıştırıcı kayıp ağırlığı (0-1, zamansal eğitim gerektirir)",
        },
        "temporal_consistency_weight": {
            "label": "Zamansal Tutarlılık Ağırlığı",
            "info": "Kare-kare benzerlik kaybı (0-5, titreşimi azaltır)",
        },
        "precision": {
            "label": "Hassasiyet",
            "choices": {
                "32": "FP32 (Standart)",
                "16-mixed": "FP16 Karışık (Daha Hızlı)",
                "bf16-mixed": "BF16 Karışık (Yeni GPU'lar)",
            },
        },
        "model_type": {
            "label": "Model Tipi",
            "info": "standard=ConvNeXt, diffusion=SD VAE hibrit, eg3d=3D-bilinçli",
            "choices": {
                "standard": "Standart (ConvNeXt)",
                "diffusion": "Difüzyon (SD VAE)",
                "eg3d": "EG3D (3D Bilinçli)",
            },
        },
        "texture_weight": {
            "label": "Doku Ağırlığı",
            "info": "Doku tutarlılık kaybı (difüzyon modeli için)",
        },
        "use_pretrained_vae": {
            "label": "Önceden Eğitilmiş VAE Kullan",
            "info": "SD VAE kullan (diffusers paketi gerektirir)",
        },
        "uniform_yaw": {
            "label": "Dengeli Yaw (Açı)",
            "info": "Eğitim örneklerini farklı yüz açılarında dengele",
        },
        "masked_training": {
            "label": "Maskeli Eğitim",
            "info": "Sadece yüz alanına odaklan (arkaplanı bulanıklaştır)",
        },
        "resume_ckpt": {
            "label": "Checkpoint'tan Devam Et",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
        },
        "refresh_preview": "🔄 Önizlemeyi Yenile",
        "preset": {
            "label": "Eğitim Ön Ayarı",
            "load": "Yükle",
            "save": "Farklı Kaydet...",
            "name_input": "Ön Ayar Adı",
            "confirm_save": "Ön Ayarı Kaydet",
            "saved": "Ön ayar kaydedildi: {name}",
            "deleted": "Ön ayar silindi: {name}",
            "load_error": "Ön ayar yüklenemedi",
        },
    },
    # Çıkarım sekmesi
    "inference": {
        "title": "Yüz Değiştirme Çıkarımı",
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
        },
        "load_model": "Modeli Yükle",
        "unload_model": "Modeli Kaldır",
        "model_status": {
            "label": "Model Durumu",
        },
        "source_image": {
            "label": "Kaynak Yüz",
        },
        "target_image": {
            "label": "Hedef Yüz",
        },
        "output_image": {
            "label": "Sonuç",
        },
        "swap": "Yüz Değiştir",
    },
    # Çıkarma sekmesi
    "extract": {
        "title": "Yüz Çıkarma",
        "description": "Eğitim için görüntülerden veya videolardan yüzleri çıkarın.",
        "input_path": {
            "label": "Girdi (görüntü, video veya dizin)",
            "placeholder": "./input_video.mp4",
        },
        "output_dir": {
            "label": "Çıktı Dizini",
        },
        "face_type": {
            "label": "Yüz Tipi",
            "choices": {
                "whole_face": "Tam Yüz",
                "full": "Full",
                "mid_full": "Orta Full",
                "half": "Yarım",
                "head": "Kafa",
            },
        },
        "output_size": {
            "label": "Çıktı Boyutu",
        },
        "min_confidence": {
            "label": "Minimum Güven",
        },
        "start": "Yüzleri Çıkar",
        "log": {
            "label": "Çıkarma Logu",
        },
        "preview": {
            "title": "Çıkarma Önizlemesi",
            "show_mask": "Maske Katmanını Göster",
            "show_mask_info": "Yüz maskesini yarı saydam katman olarak göster",
            "last_face": "Son Çıkarılan Yüz",
            "face_info": "Yüz Bilgisi",
            "gallery": {"label": "Çıkarılan Yüzler"},
        },
        "status": {"label": "Durum"},
    },
    # Yüz Seti Tarayıcı
    "faceset_browser": {
        "title": "Yüz Seti Tarayıcı",
        "directory": "Dizin",
        "load": "Yükle",
        "refresh": "Yenile",
        "faces": "Yüzler",
        "show_masks": "Maskeleri Göster",
        "sort_by": "Sıralama",
        "page_size": "Sayfa Boyutu",
        "prev": "<< Önceki",
        "next": "Sonraki >>",
        "delete_selected": "Seçilileri Sil",
        "clear_selection": "Seçimi Temizle",
        "undo_last_delete": "Son Silmeyi Geri Al",
        "selected": "Seçili Yüz",
        "metadata": "Meta Veri",
        "no_directory": "Dizin belirtilmedi",
        "not_found": "Dizin bulunamadı",
        "status_loaded": "{count} yüz yüklendi",
        "status_selected": "Seçili: {count}",
        "status_load_errors": "Sayfada yükleme hatası: {count}",
        "no_selected_files": "Seçili dosya yok",
        "selection_cleared": "Seçim temizlendi",
        "trash_summary": "Batch {batch_id}: taşınan {moved}, eksik {missing}, hatalı {failed}",
        "undo_summary": "Geri al {batch_id}: geri yüklenen {restored}, atlanan {skipped}, hatalı {failed}",
        "no_trash_batch": "Geri alınacak trash batch yok",
    },
    # Ayarlar sekmesi
    "settings": {
        "title": "Ayarlar",
        "device_section": "Cihaz Yapılandırması",
        "performance_section": "Performans",
        "language_section": "Dil",
        "device": {
            "label": "Hesaplama Cihazı",
            "info": "Model çıkarımı ve eğitimi için cihaz seçin",
            "choices": {
                "auto": "Otomatik (En İyiyi Algıla)",
                "cuda": "CUDA (GPU)",
                "cpu": "CPU",
            },
        },
        "batch_size": {
            "label": "Varsayılan Batch Boyutu",
            "info": "Aynı anda işlenen örnek sayısı",
        },
        "num_workers": {
            "label": "Worker Sayısı",
            "info": "Veri yükleme için worker thread sayısı (0 = sadece ana thread)",
        },
        "locale": {
            "label": "Dil",
            "info": "Uygulama görüntüleme dili",
            "choices": {
                "en": "İngilizce",
                "tr": "Türkçe",
            },
        },
        "status": {
            "label": "Durum",
            "saved": "Ayarlar başarıyla kaydedildi",
            "saved_reload": "Ayarlar kaydedildi. Dil değişikliğini uygulamak için uygulamayı yeniden başlatın.",
        },
    },
    # Birleştirme sekmesi
    "merge": {
        "title": "Video Yüz Değiştirme",
        "description": "Özelleştirilebilir harmanlama ve renk transferi ile eğitilmiş modelleri kullanarak videoları işleyin.",
        "input_video": {
            "label": "Girdi Videosu",
            "placeholder": "./input.mp4",
            "info": "Kaynak video dosyasının yolu",
        },
        "output_video": {
            "label": "Çıktı Videosu",
            "placeholder": "./output.mp4",
            "info": "İşlenmiş video çıktısının yolu",
        },
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Eğitilmiş model checkpoint yolu",
        },
        "color_transfer": {
            "label": "Renk Transferi Modu",
            "info": "RCT=Reinhard, LCT=Lineer, SOT/MKL/IDT/MIX/Hist-Match desteklenir",
            "choices": {
                "none": "Yok",
                "rct": "RCT (Reinhard)",
                "lct": "LCT (Lineer)",
                "sot": "SOT (Dilimli OT)",
                "mkl": "MKL (Monge-Kantorovitch)",
                "idt": "IDT (İteratif)",
                "mix": "Mix (LCT+SOT En İyi)",
                "hist-match": "Histogram Eşleştirme",
            },
        },
        "blend_mode": {
            "label": "Harmanlama Modu",
            "info": "Laplacian=piramit, Poisson=kesintisiz, Feather=alfa",
            "choices": {
                "laplacian": "Laplacian (Piramit)",
                "poisson": "Poisson (Kesintisiz)",
                "feather": "Feather (Alfa)",
            },
        },
        "restoration": {
            "title": "Yüz Restorasyon",
            "enable": {
                "label": "GFPGAN Etkinleştir",
                "info": "GFPGAN ile yüz kalitesini artır",
            },
            "strength": {
                "label": "Restorasyon Gücü",
            },
            "version": {
                "label": "GFPGAN Sürümü",
            },
        },
        "encoding": {
            "title": "Video Kodlama",
            "codec": {
                "label": "Kodlayıcı",
                "info": "'auto' mevcutsa NVENC seçer",
                "choices": {
                    "auto": "Otomatik (En İyi Mevcut)",
                    "libx264": "libx264 (CPU H.264)",
                    "libx265": "libx265 (CPU H.265)",
                    "h264_nvenc": "NVENC H.264 (GPU)",
                    "hevc_nvenc": "NVENC H.265 (GPU)",
                },
            },
            "crf": {
                "label": "Kalite (CRF)",
                "info": "Düşük = daha iyi kalite, daha büyük dosya",
            },
        },
        "start": "Birleştirmeyi Başlat",
        "stop": "Birleştirmeyi Durdur",
        "log": {
            "label": "Birleştirme Logu",
        },
    },
    # İnteraktif Birleştirme sekmesi
    "interactive_merge": {
        "title": "İnteraktif Birleştirme",
        "description": "Ayarlanabilir parametrelerle gerçek zamanlı önizleme. Başlamak için eğitilmiş bir model ve kare dizisi yükleyin.",
        # Oturum kurulumu
        "session": {
            "title": "Oturum Kurulumu",
        },
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Eğitilmiş model checkpoint yolu",
        },
        "frames_dir": {
            "label": "Kareler Dizini",
            "placeholder": "./frames",
            "info": "Girdi kare görüntülerini içeren dizin",
        },
        "output_dir": {
            "label": "Çıktı Dizini",
            "placeholder": "./output",
            "info": "Dışa aktarılan kareler için dizin",
        },
        "load_session": "Oturumu Yükle",
        "session_status": {
            "label": "Oturum Durumu",
            "not_loaded": "Oturum yüklenmedi",
            "loaded": "Yüklendi: {path} konumundan {count} kare",
        },
        # Ayarlar
        "settings": {
            "title": "Birleştirme Ayarları",
        },
        "mode": {
            "label": "Birleştirme Modu",
            "info": "Değiştirilen yüzün nasıl harmanlanacağı",
            "choices": {
                "original": "Orijinal",
                "overlay": "Kaplama",
                "hist-match": "Histogram Eşleştirme",
                "seamless": "Kesintisiz",
                "seamless-hist-match": "Kesintisiz + Histogram",
            },
        },
        "mask_mode": {
            "label": "Maske Modu",
            "info": "Yüz bölgesi segmentasyon yöntemi",
            "choices": {
                "full": "Tam",
                "convex_hull": "Dışbükey Örtü",
                "dst": "Hedef",
                "segmented": "Segmentli",
            },
        },
        "color_transfer": {
            "label": "Renk Transferi",
            "info": "Renk eşleştirme algoritması",
            "choices": {
                "none": "Yok",
                "rct": "RCT (Reinhard)",
                "lct": "LCT (Lineer)",
                "mkl": "MKL (Monge-Kantorovitch)",
                "idt": "IDT (İteratif)",
                "sot": "SOT (Dilimli OT)",
                "mix": "Mix (LCT+SOT En İyi)",
                "hist-match": "Histogram Eşleştirme",
            },
        },
        # Maske işleme
        "mask_processing": {
            "title": "Maske İşleme",
        },
        "erode_mask": {
            "label": "Maskeyi Aşındır",
            "info": "Negatif = genişlet, Pozitif = aşındır",
        },
        "blur_mask": {
            "label": "Maskeyi Bulanıklaştır",
            "info": "Maske kenar yumuşaklığı",
        },
        "face_scale": {
            "label": "Yüz Ölçeği",
            "info": "Değiştirilen yüz boyutunu ayarla",
        },
        # Keskinleştirme
        "sharpening": {
            "title": "Keskinleştirme",
        },
        "sharpen_mode": {
            "label": "Keskinleştirme Modu",
            "choices": {
                "none": "Yok",
                "box": "Kutu",
                "gaussian": "Gauss",
            },
        },
        "sharpen_amount": {
            "label": "Keskinleştirme Miktarı",
            "info": "Negatif = bulanıklaştır, Pozitif = keskinleştir",
        },
        # Gelişmiş
        "advanced": {
            "title": "Gelişmiş",
        },
        "hist_threshold": {
            "label": "Histogram Eşleştirme Eşiği",
        },
        "restore_face": {
            "label": "GFPGAN Restorasyon",
            "info": "Yüz kalitesini artır",
        },
        "restore_strength": {
            "label": "Restorasyon Gücü",
        },
        # Süper Çözünürlük
        "super_resolution": {
            "title": "Süper Çözünürlük",
        },
        "super_resolution_power": {
            "label": "Süper Çözünürlük Gücü",
            "info": "4x büyütme karışımı (0 = devre dışı, 100 = tam iyileştirme)",
        },
        # Uygula
        "apply_settings": "Ayarları Uygula",
        "config_status": {
            "label": "Geçerli Yapılandırma",
            "showing_original": "Orijinal gösteriliyor (ayarlar uygulanmadı)",
        },
        # Önizleme
        "preview": {
            "title": "Önizleme",
            "image_label": "Önizleme",
        },
        "show_original": {
            "label": "Orijinali Göster",
        },
        "frame_info": {
            "label": "Kare Bilgisi",
            "no_frame": "Kare yüklenmedi",
            "format": "Kare {current}/{total}",
            "detail": "Kare {current}/{total}: {filename}",
        },
        # Navigasyon
        "nav": {
            "prev": "◀ Önceki",
            "next": "Sonraki ▶",
            "frame": "Kare",
        },
        # Dışa aktarma
        "export": {
            "title": "Dışa Aktar",
            "current": "Geçerli Kareyi Dışa Aktar",
            "all": "Tüm Kareleri Dışa Aktar",
            "save_session": "Oturumu Kaydet",
            "status_label": "Dışa Aktarma Durumu",
            "current_success": "Şuraya dışa aktarıldı: {path}",
            "all_success": "{count} kare dışa aktarıldı",
            "session_saved": "Oturum şuraya kaydedildi: {path}",
            "failed": "Dışa aktarma başarısız",
            "save_failed": "Kaydetme başarısız",
        },
        # Hatalar
        "errors": {
            "no_session": "Oturum yüklenmedi",
            "load_failed": "Oturum yüklenirken hata",
            "update_failed": "Yapılandırma güncellenirken hata",
            "navigate_failed": "Navigasyon hatası",
        },
    },
    # Sıralama sekmesi
    "sort": {
        "title": "Veri Seti Sıralama",
        "description": "Yüz görüntülerini çeşitli kriterlere göre sıralayın ve filtreleyin.",
        "input_dir": {
            "label": "Girdi Dizini",
            "placeholder": "./workspace/data_src/aligned",
            "info": "Hizalanmış yüz görüntülerini içeren dizin",
        },
        "output_dir": {
            "label": "Çıktı Dizini (isteğe bağlı)",
            "placeholder": "Yerinde sıralamak için boş bırakın",
            "info": "Sıralanmış görüntüler için isteğe bağlı çıktı dizini",
        },
        "method": {
            "label": "Sıralama Yöntemi",
            "info": "Sıralama/filtreleme algoritmasını seçin",
            "choices": {
                "blur": "Bulanıklık (Keskinlik)",
                "blur-fast": "Bulanıklık Hızlı (Laplacian)",
                "motion-blur": "Hareket Bulanıklığı",
                "face-yaw": "Yüz Yaw (Sol-Sağ)",
                "face-pitch": "Yüz Pitch (Yukarı-Aşağı)",
                "face-source-rect-size": "Yüz Boyutu",
                "hist": "Histogram Benzerliği",
                "hist-dissim": "Histogram Farklılığı",
                "absdiff": "Mutlak Fark",
                "absdiff-dissim": "Mutlak Fark Farklılık",
                "id-sim": "Kimlik Benzerliği",
                "id-dissim": "Kimlik Farklılığı",
                "ssim": "SSIM Benzerliği",
                "ssim-dissim": "SSIM Farklılığı",
                "brightness": "Parlaklık",
                "hue": "Ton",
                "black": "Siyah Pikseller",
                "origname": "Orijinal İsim",
                "oneface": "Sadece Tek Yüz",
                "final": "Final (En İyi Seçim)",
                "final-fast": "Final Hızlı",
            },
        },
        "exec_mode": {
            "label": "Çalıştırma Modu",
            "info": "Sıralama yükleri için paralel backend",
            "choices": {
                "auto": "Otomatik",
                "process": "Process Havuzu",
                "thread": "Thread Havuzu",
            },
        },
        "exact_limit": {
            "label": "Kesin Hesap Sınırı",
            "info": "0 = yöntem varsayılanı. Yüksek değerler küçük sette O(n^2) kesin yolu açar",
        },
        "jobs": {
            "label": "Paralel Worker",
            "info": "0 = otomatik (CPU sayısı), aksi halde sabit worker",
        },
        "target_count": {
            "label": "Hedef Sayı",
            "info": "Sadece 'final' ve 'final-fast' yöntemleri için kullanılır",
        },
        "dry_run": {
            "label": "Kuru Çalıştırma (Önizleme)",
            "info": "Değişiklik yapmadan ne olacağını göster",
        },
        "start": "Sıralamayı Başlat",
        "stop": "Sıralamayı Durdur",
        "log": {
            "label": "Sıralama Logu",
        },
    },
    # Dışa aktarma sekmesi
    "export": {
        "title": "Model Dışa Aktarma",
        "description": "Optimize edilmiş çıkarım için eğitilmiş modelleri ONNX veya TensorRT'ye dışa aktarın.",
        "input_path": {
            "label": "Girdi Yolu",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "ONNX için Checkpoint (.ckpt) veya TensorRT için ONNX (.onnx)",
        },
        "output_path": {
            "label": "Çıktı Yolu",
            "placeholder": "./model.onnx",
            "info": "Çıktı dosya yolu (.onnx veya .engine)",
        },
        "format": {
            "label": "Dışa Aktarma Formatı",
            "info": "ONNX çapraz platform için, TensorRT NVIDIA GPU'lar için",
            "choices": {
                "onnx": "ONNX",
                "tensorrt": "TensorRT",
            },
        },
        "precision": {
            "label": "Hassasiyet",
            "info": "Hız ve kalite dengesi için FP16 önerilir",
            "choices": {
                "fp32": "FP32 (Tam Hassasiyet)",
                "fp16": "FP16 (Yarı Hassasiyet)",
                "int8": "INT8 (Kuantize)",
            },
        },
        "validate": {
            "label": "Dışa Aktarmayı Doğrula",
            "info": "Dışa aktarılan modeli PyTorch orijinaliyle karşılaştır",
        },
        "start": "Modeli Dışa Aktar",
        "stop": "Dışa Aktarmayı Durdur",
        "log": {
            "label": "Dışa Aktarma Logu",
        },
    },
    # Video Araçları sekmesi
    "video_tools": {
        "title": "Video Araçları",
        "description": "Video-kare ve kare-video dönüşüm araçları.",
        "extract": {
            "title": "Videodan Kare Çıkar",
            "input": {
                "label": "Girdi Videosu",
                "placeholder": "./input.mp4",
                "info": "Video dosyasının yolu",
            },
            "output": {
                "label": "Çıktı Dizini",
                "placeholder": "./frames",
                "info": "Çıkarılan karelerin kaydedileceği dizin",
            },
            "fps": {
                "label": "FPS (0 = orijinal)",
                "info": "Hedef kare hızı (orijinali korumak için 0)",
            },
            "format": {
                "label": "Çıktı Formatı",
            },
            "start": "Kareleri Çıkar",
            "log": {
                "label": "Log",
            },
        },
        "create": {
            "title": "Karelerden Video Oluştur",
            "input": {
                "label": "Girdi Dizini",
                "placeholder": "./frames",
                "info": "Görüntü dizisini içeren dizin",
            },
            "output": {
                "label": "Çıktı Videosu",
                "placeholder": "./output.mp4",
                "info": "Çıktı video yolu",
            },
            "fps": {
                "label": "FPS",
            },
            "codec": {
                "label": "Kodek",
            },
            "bitrate": {
                "label": "Bit Hızı",
                "info": "Video bit hızı (örn. 16M, 25M)",
            },
            "start": "Video Oluştur",
            "log": {
                "label": "Log",
            },
        },
        "cut": {
            "title": "Video Segmenti Kes",
            "input": {
                "label": "Girdi Videosu",
                "placeholder": "./input.mp4",
            },
            "output": {
                "label": "Çıktı Videosu",
                "placeholder": "./cut_output.mp4",
            },
            "start_time": {
                "label": "Başlangıç Zamanı",
                "info": "Format: SS:DD:SS veya saniye",
            },
            "end_time": {
                "label": "Bitiş Zamanı",
                "info": "Format: SS:DD:SS veya saniye",
            },
            "codec": {
                "label": "Kodek",
                "info": "Akış kopyalama için 'copy' kullanın veya yeniden kodlama kodeği seçin",
            },
            "audio_track": {
                "label": "Ses Kanalı ID",
                "info": "Korunacak ses akışı indeksi (varsayılan: 0)",
            },
            "bitrate": {
                "label": "Bit Hızı (isteğe bağlı)",
                "info": "Yalnızca kodek copy değilse kullanılır (örn. 16M)",
            },
            "start": "Videoyu Kes",
            "log": {
                "label": "Log",
            },
        },
        "denoise": {
            "title": "Zamansal Gürültü Azaltma",
            "description": "Kare dizilerinde titreşimi azaltmak için zamansal gürültü azaltma uygulayın.",
            "input": {
                "label": "Girdi Dizini",
                "placeholder": "./frames",
                "info": "Görüntü dizisini içeren dizin",
            },
            "output": {
                "label": "Çıktı Dizini (isteğe bağlı)",
                "placeholder": "Yerinde işlem için boş bırakın",
            },
            "factor": {
                "label": "Gürültü Azaltma Faktörü",
                "info": "Zamansal pencere boyutu (tek sayı olmalı)",
            },
            "start": "Gürültü Azaltmayı Uygula",
            "log": {
                "label": "Log",
            },
        },
    },
    # Yüz Seti Araçları sekmesi
    "faceset_tools": {
        "title": "Yüz Seti Araçları",
        "description": "Yüz veri setlerini iyileştirme ve yeniden boyutlandırma araçları.",
        "enhance": {
            "title": "Yüz İyileştirme (GFPGAN)",
            "description": "GFPGAN restorasyon kullanarak yüz kalitesini iyileştirin.",
            "input": {
                "label": "Girdi Dizini",
                "placeholder": "./workspace/data_src/aligned",
                "info": "Yüz görüntülerini içeren dizin",
            },
            "output": {
                "label": "Çıktı Dizini (isteğe bağlı)",
                "placeholder": "Otomatik adlandırma için boş bırakın",
                "info": "Çıktı dizini (varsayılan: input_enhanced)",
            },
            "strength": {
                "label": "İyileştirme Gücü",
                "info": "0 = orijinal, 1 = tam iyileştirilmiş",
            },
            "model": {
                "label": "GFPGAN Sürümü",
            },
            "start": "Yüz Setini İyileştir",
            "log": {
                "label": "Log",
            },
        },
        "resize": {
            "title": "Yüz Seti Yeniden Boyutlandırma",
            "description": "DFL meta veri koruması ile yüz görüntülerini yeniden boyutlandırın.",
            "input": {
                "label": "Girdi Dizini",
                "placeholder": "./workspace/data_src/aligned",
                "info": "Yüz görüntülerini içeren dizin",
            },
            "output": {
                "label": "Çıktı Dizini (isteğe bağlı)",
                "placeholder": "Otomatik adlandırma için boş bırakın",
                "info": "Çıktı dizini (varsayılan: input_SIZE)",
            },
            "size": {
                "label": "Hedef Boyut",
                "info": "Çıktı görüntü boyutu (genişlik = yükseklik)",
            },
            "face_type": {
                "label": "Yüz Tipi",
                "info": "Hedef yüz tipi (keep = orijinali koru)",
            },
            "interp": {
                "label": "İnterpolasyon",
            },
            "start": "Yüz Setini Yeniden Boyutlandır",
            "log": {
                "label": "Log",
            },
        },
    },
    # Toplu işleme sekmesi
    "batch": {
        "title": "Toplu İşleme",
        "description": "Birden fazla videoyu aynı ayarlarla sırayla işleyin.",
        "files": {
            "label": "Videoları Seç",
        },
        "output_dir": {
            "label": "Çıktı Dizini",
            "placeholder": "./batch_output",
            "info": "İşlenmiş video çıktıları için dizin",
        },
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Eğitilmiş model checkpoint yolu",
        },
        "operation": {
            "label": "İşlem",
        },
        "add_to_queue": "Kuyruğa Ekle",
        "queue": {
            "title": "İşlem Kuyruğu",
            "file": "Dosya",
            "status": "Durum",
            "progress": "İlerleme",
        },
        "progress": {
            "label": "Genel İlerleme",
        },
        "start_all": "Tümünü Başlat",
        "stop_all": "Tümünü Durdur",
        "clear_completed": "Tamamlananları Temizle",
        "status": {
            "label": "Durum",
            "no_files": "Dosya seçilmedi",
            "added": "{count} dosya kuyruğa eklendi",
            "started": "{count} öğe işlemeye başlandı",
            "stopped": "İşleme durduruldu",
            "cleared": "{count} tamamlanan öğe temizlendi",
            "already_running": "Toplu işleme zaten çalışıyor",
            "no_pending": "Kuyrukta bekleyen öğe yok",
        },
    },
    # Model karşılaştırma sekmesi
    "compare": {
        "title": "Model Karşılaştırma",
        "description": "İki farklı model checkpoint'ının çıktılarını yan yana karşılaştırın.",
        "checkpoint_a": {
            "label": "Model A Checkpoint",
            "placeholder": "./model_a.ckpt",
        },
        "checkpoint_b": {
            "label": "Model B Checkpoint",
            "placeholder": "./model_b.ckpt",
        },
        "load_model_a": "Model A Yükle",
        "load_model_b": "Model B Yükle",
        "status_a": {"label": "Model A Durumu"},
        "status_b": {"label": "Model B Durumu"},
        "test_image": {"label": "Test Görüntüsü"},
        "compare": "Modelleri Karşılaştır",
        "results": {"title": "Karşılaştırma Sonuçları"},
        "metrics": {"label": "Kalite Metrikleri (SSIM/PSNR)"},
        "unload_all": "Tüm Modelleri Kaldır",
        "errors": {
            "both_models_required": "Karşılaştırmadan önce her iki modeli de yükleyin",
        },
    },
    # İşlem Sonrası sekmesi
    "postprocess": {
        "title": "İşlem Sonrası",
        "color_transfer": {
            "title": "Renk Transferi Demosu",
            "apply": "Renk Transferi Uygula",
        },
        "blending": {
            "title": "Harmanlama Demosu",
            "apply": "Görüntüleri Harmanla",
        },
        "restoration": {
            "title": "Yüz Restorasyon Demosu",
            "description": "GFPGAN veya GPEN kullanarak yüz kalitesini iyileştirin.",
            "apply": "Yüzü Restore Et",
        },
        "neural": {
            "title": "Sinirsel Renk Transferi",
            "description": "Daha gerçekçi sonuçlar için VGG tabanlı semantik renk eşleştirme.",
            "apply": "Sinirsel Renk Uygula",
        },
        "ct_source": {"label": "Kaynak (renk referansı)"},
        "ct_target": {"label": "Hedef (değiştirilecek)"},
        "ct_result": {"label": "Sonuç"},
        "ct_mode": {
            "label": "Renk Transferi Modu",
            "info": "RCT=Reinhard, LCT=Lineer, SOT=Dilimli OT, MKL=Monge-Kantorovitch, IDT=İteratif",
        },
        "bl_fg": {"label": "Ön Plan"},
        "bl_bg": {"label": "Arka Plan"},
        "bl_mask": {"label": "Maske"},
        "bl_result": {"label": "Sonuç"},
        "bl_mode": {
            "label": "Harmanlama Modu",
            "info": "Laplacian=çok bantlı piramit, Poisson=kesintisiz klonlama, Feather=alfa harmanlama",
        },
        "restore_input": {"label": "Girdi Yüzü"},
        "restore_result": {"label": "Restore Edilmiş Yüz"},
        "restore_mode": {
            "label": "Restorasyon Modu",
            "info": "GFPGAN: En iyi kalite, GPEN: Daha iyi yapı koruması",
        },
        "restore_strength": {
            "label": "Restorasyon Gücü",
            "info": "0 = orijinal, 1 = tam restore",
        },
        "restore_version": {
            "label": "GFPGAN Sürümü",
            "info": "Sadece mod GFPGAN olduğunda kullanılır",
        },
        "gpen_size": {
            "label": "GPEN Model Boyutu",
            "info": "Sadece mod GPEN olduğunda kullanılır. Büyük = daha iyi kalite, daha yavaş",
        },
        "nct_source": {"label": "Stil Referansı (renk kaynağı)"},
        "nct_target": {"label": "Hedef Görüntü (değiştirilecek)"},
        "nct_result": {"label": "Sonuç"},
        "nct_mode": {
            "label": "Transfer Modu",
            "info": "histogram=LAB uzayı, statistics=ortalama/std, gram=stil (torchvision gerektirir)",
        },
        "nct_strength": {"label": "Transfer Gücü"},
        "nct_preserve": {"label": "Parlaklığı Koru"},
    },
    # Maske Düzenleyici sekmesi
    "mask_editor": {
        "title": "Maske Düzenleyici",
        "description": "LoRA ince ayar ile yüz segmentasyon maskelerini düzenleyin",
        "tabs": {
            "editor": "Maskeleri Düzenle",
            "training": "LoRA Eğitimi",
            "batch": "Toplu Uygula",
        },
        "components": {
            "title": "Yüz Bileşenleri",
        },
        "canvas": {
            "title": "Maske Kanvası",
            "label": "Çiz/Sil Maske",
            "preview": "Önizleme",
            "mask_only": "Sadece Maske",
            "rebuild": "Maskeyi Yeniden Oluştur",
            "reset": "Sıfırla",
        },
        "refine": {
            "title": "İyileştirme",
            "erode": "Aşındır",
            "dilate": "Genişlet",
            "blur": "Bulanıklaştır",
        },
        "editor": {
            "faceset_dir": "Yüz Seti Dizini",
            "faces": "Yüzler",
            "selected": "Seçili",
            "samples_count": "Eğitim Örnekleri",
            "status": "Durum",
        },
        "actions": {
            "save_mask": "Maskeyi Görüntüye Kaydet",
            "save_sample": "Eğitim Setine Ekle",
        },
        "lora": {
            "title": "LoRA İnce Ayar",
            "samples_dir": "Örnekler Dizini",
            "output_dir": "Çıktı Dizini",
            "epochs": "Epoch",
            "rank": "LoRA Rank",
            "learning_rate": "Öğrenme Oranı",
            "start": "Eğitimi Başlat",
            "stop": "Eğitimi Durdur",
            "progress": "İlerleme",
            "log": "Eğitim Logu",
        },
        "batch": {
            "title": "Toplu Uygula",
            "input_dir": "Girdi Dizini",
            "output_dir": "Çıktı Dizini",
            "use_lora": "LoRA Adaptörü Kullan",
            "lora_weights": "LoRA Ağırlık Dosyası",
            "components": "Maske Bileşenleri",
            "refinement": "İyileştirme Ayarları",
            "preview_before_save": "Kaydetmeden önce önizle",
            "apply": "Yüz Setine Uygula",
            "confirm": "Onayla ve Tümünü Kaydet",
            "cancel": "İptal",
            "preview_gallery": "Önizleme",
            "progress": "İlerleme",
        },
    },
    # Wizard sekmesi
    "wizard": {
        "title": "Hızlı Başlangıç Sihirbazı",
        "description": "İlk yüz değiştirme videonuzu oluşturmak için adım adım rehber. Yeni başlayanlar için mükemmel!",
        "next_step": "Sonraki Adım →",
        "back": "← Geri",
        "steps": {
            "upload": "Yükle",
            "extract": "Çıkar",
            "train": "Eğit",
            "apply": "Uygula",
        },
        "step1": {
            "title": "Adım 1: Video Yükle",
            "description": "Kaynak video (kopyalanacak yüz) ve hedef video (değiştirilecek yüz) yükleyin.",
            "src_video": "Kaynak Video (kopyalanacak yüz)",
            "dst_video": "Hedef Video (değiştirilecek yüz)",
            "status": "Durum",
            "error_missing_videos": "Lütfen hem kaynak hem de hedef videoları yükleyin.",
            "success": "Videolar başarıyla yüklendi! Çıkarma adımına geçin.",
        },
        "step2": {
            "title": "Adım 2: Yüzleri Çıkar",
            "description": "Her iki videodan yüzleri çıkarın ve hizalayın. Bu birkaç dakika sürebilir.",
            "face_type": "Yüz Tipi",
            "output_size": "Çıktı Boyutu",
            "log": "Çıkarma Logu",
            "extract": "Çıkarmayı Başlat",
        },
        "step3": {
            "title": "Adım 3: Modeli Eğit",
            "description": "Yüz değiştirme modelini eğitin. Ayarlara bağlı olarak 30-60 dakika sürebilir.",
            "epochs": "Eğitim Epoch Sayısı",
            "batch_size": "Batch Boyutu",
            "preset": "Eğitim Ön Ayarı",
            "log": "Eğitim Logu",
            "preview": "Eğitim Önizleme",
            "train": "Eğitimi Başlat",
            "stop": "Eğitimi Durdur",
            "stopped": "Eğitim kullanıcı tarafından durduruldu.",
        },
        "step4": {
            "title": "Adım 4: Videoya Uygula",
            "description": "Eğitilmiş modeli uygulayarak son yüz değiştirme videosunu oluşturun.",
            "color_transfer": "Renk Transferi",
            "blend_mode": "Harmanlama Modu",
            "log": "Birleştirme Logu",
            "apply": "Video Oluştur",
            "status": "Son Durum",
            "result": "Sonuç Videosu",
            "error_no_model": "Eğitilmiş model bulunamadı. Lütfen önce eğitimi tamamlayın.",
            "success": "Video başarıyla oluşturuldu: {path}",
        },
    },
}
