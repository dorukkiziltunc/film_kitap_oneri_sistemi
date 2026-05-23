# Film ve Kitap Öneri Sistemi

## 📌 Projenin Amacı ve Özeti
Bu proje, veri analizi ve makine öğrenimi temellerini kullanarak, kullanıcıların geçmiş puanlama davranışlarına göre yeni içerik (film/kitap) önerileri sunan etkileşimli bir sistemdir. Sistem, terminal tabanlı bir arayüz ile yönetilmektedir.

## ⚙️ Uygulanan Yöntemler ve Karşılaştırma
Bu projede istenen hedeflere ulaşılmış ve iki farklı yaklaşım aynı sistemde karşılaştırılmıştır:

1. **User-Based Collaborative Filtering:** Hedef kullanıcı ile diğer kullanıcılar arasındaki Cosine Similarity (Kosinüs Benzerliği) hesaplanarak zevk ikizleri bulunur.
2. **Item-Based Collaborative Filtering:** Kullanıcıların tercih ettiği içerikler ile diğer içerikler arasındaki benzerlik matrisi kurularak, içeriğe dayalı öneri sunulur.

## 🎯 Gerçekleştirilen Başarımlar
- [x] Veri seti analiz raporu fonksiyonu eklendi (Kullanıcı/İçerik sayısı, Ort. Puan vb.).
- [x] Konsol tabanlı etkileşimli kullanıcı arayüzü (Terminal Menü) tasarlandı.
- [x] Kullanıcıdan çalışma anında dinamik olarak `user_id` girişi alındı.
- [x] Sadece film, sadece kitap veya özel kategori bazında **filtreleme yeteneği** eklendi.
- [x] Çıktı formatı *[Sıra | Önerilen İçerik | Tür | Skor]* formatına ayarlandı.
- [x] Öneriler `outputs/recommendations.csv` dosyasına dinamik olarak aktarıldı.

## 🛠️ Kurulum ve Çalıştırma
```bash
# Kütüphaneleri yükleyin
pip install pandas numpy scikit-learn

# Projeyi çalıştırın
python src/main.py
