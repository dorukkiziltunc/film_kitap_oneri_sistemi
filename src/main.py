import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')  # Konsolun temiz görünmesi için uyarıları gizler


def load_data():
    """Veri setlerini okur ve birleştirir."""
    ratings = pd.read_csv('../data/ratings.csv')
    items = pd.read_csv('../data/items.csv')
    df = pd.merge(ratings, items, on='item_id')
    return df


def analyze_data(df):
    """Veri seti hakkında kısa analiz raporu sunar. (Bölüm 8 Beklentisi)"""
    print("\n" + "=" * 40)
    print("📊 VERİ SETİ ANALİZ RAPORU")
    print("=" * 40)
    print(f"Toplam Kullanıcı Sayısı : {df['user_id'].nunique()}")
    print(f"Toplam İçerik Sayısı    : {df['item_id'].nunique()}")
    print(f"Toplam Verilen Puan     : {len(df)}")
    print(f"Genel Ortalama Puan     : {round(df['rating'].mean(), 2)}")

    print("\nEn Popüler Türler (Kategori Dağılımı):")
    populer_turler = df['category'].value_counts()
    for tur, sayi in populer_turler.items():
        print(f"- {tur}: {sayi} değerlendirme")
    print("=" * 40 + "\n")


def create_matrices(df):
    """Kullanıcı ve İçerik matrislerini oluşturup benzerliklerini hesaplar."""
    # 1. Kullanıcı - İçerik Matrisi
    user_item_matrix = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

    # 2. Kullanıcı Benzerlik Matrisi (User-Based için)
    user_sim_matrix = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    # 3. İçerik Benzerlik Matrisi (Item-Based için - Bonus)
    item_sim_matrix = cosine_similarity(user_item_matrix.T)
    item_sim_df = pd.DataFrame(item_sim_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    return user_item_matrix, user_sim_df, item_sim_df


def get_recommendations(target_user, user_item_matrix, sim_df, df, method="user", top_n=5, filter_type=None,
                        filter_category=None):
    """Filtrelenebilir dinamik öneri motoru."""
    if target_user not in user_item_matrix.index:
        return pd.DataFrame()  # Kullanıcı yoksa boş döndür

    user_ratings = user_item_matrix.loc[target_user]
    unrated_items = user_ratings[user_ratings == 0].index.tolist()
    rated_items = user_ratings[user_ratings > 0]

    recommendations = {}

    if method == "user":
        # USER-BASED YAKLAŞIMI
        similar_users = sim_df[target_user].drop(target_user)
        for item in unrated_items:
            toplam_skor, benzerlik_toplami = 0, 0
            for sim_user, sim_score in similar_users.items():
                rating = user_item_matrix.loc[sim_user, item]
                if rating > 0:
                    toplam_skor += sim_score * rating
                    benzerlik_toplami += sim_score
            if benzerlik_toplami > 0:
                recommendations[item] = toplam_skor / benzerlik_toplami

    elif method == "item":
        # ITEM-BASED YAKLAŞIMI (Bonus Geliştirme)
        for unrated_item in unrated_items:
            toplam_skor, benzerlik_toplami = 0, 0
            for rated_item, rating in rated_items.items():
                sim_score = sim_df.loc[unrated_item, rated_item]
                if sim_score > 0:
                    toplam_skor += sim_score * rating
                    benzerlik_toplami += sim_score
            if benzerlik_toplami > 0:
                recommendations[item] = toplam_skor / benzerlik_toplami

    # Filtreleme İşlemleri (Bonus Geliştirme)
    filtered_recs = {}
    for title, score in recommendations.items():
        item_info = df[df['title'] == title].iloc[0]
        # Sadece Film veya Sadece Kitap Filtresi
        if filter_type and item_info['type'].lower() != filter_type.lower():
            continue
        # Kategori Filtresi
        if filter_category and item_info['category'].lower() != filter_category.lower():
            continue
        filtered_recs[title] = score

    # Skora göre sıralama
    sorted_recs = sorted(filtered_recs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # İstenen Tablo Formatına Dönüştürme (Bölüm 8.1 Beklentisi)
    result = []
    for rank, (title, score) in enumerate(sorted_recs, start=1):
        item_info = df[df['title'] == title].iloc[0]
        result.append({
            "Sıra": rank,
            "Önerilen İçerik": title,
            "Tür": item_info['category'],  # veya item_info['type']
            "Skor": round(score, 2)
        })

    return pd.DataFrame(result)


def interactive_menu():
    """Kullanıcı etkileşimli terminal arayüzü (Bonus Geliştirme)"""
    print("Veriler yükleniyor...")
    try:
        df = load_data()
        matrix, user_sim, item_sim = create_matrices(df)
    except FileNotFoundError:
        print("HATA: data/ratings.csv veya data/items.csv bulunamadı!")
        return

    while True:
        print("\n" + "*" * 45)
        print("🎬 FİLM VE KİTAP ÖNERİ SİSTEMİ 📚")
        print("*" * 45)
        print("1. Veri Seti Analizini Gör")
        print("2. Öneri Üret (User-Based & Item-Based Karşılaştırması)")
        print("3. Filtreli Öneri Üret (Sadece Film / Kitap / Kategori)")
        print("4. Çıkış")
        print("*" * 45)

        secim = input("Lütfen bir işlem seçin (1-4): ")

        if secim == '1':
            analyze_data(df)

        elif secim == '2':
            try:
                user_id = int(input("Öneri üretilecek Kullanıcı ID'sini girin: "))
                print("\n--- 1. YÖNTEM: USER-BASED ÖNERİLER ---")
                user_recs = get_recommendations(user_id, matrix, user_sim, df, method="user")
                if not user_recs.empty:
                    print(user_recs.to_string(index=False))
                    user_recs.to_csv('../outputs/recommendations.csv', index=False)
                    print("\n[+] Sonuçlar '../outputs/recommendations.csv' olarak kaydedildi.")
                else:
                    print("Bu kullanıcı için yeterli veri yok veya kullanıcı bulunamadı.")

                print("\n--- 2. YÖNTEM: ITEM-BASED ÖNERİLER ---")
                item_recs = get_recommendations(user_id, matrix, item_sim, df, method="item")
                if not item_recs.empty:
                    print(item_recs.to_string(index=False))
                else:
                    print("Bu kullanıcı için yeterli veri yok.")

            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        elif secim == '3':
            try:
                user_id = int(input("Kullanıcı ID'sini girin: "))
                f_type = input("Sadece 'Film' mi yoksa 'Kitap' mı? (Tümü için boş bırakın): ").strip().capitalize()
                if f_type not in ['Film', 'Kitap']: f_type = None

                f_cat = input("Özel bir kategori var mı? (Örn: Bilim Kurgu / Tümü için boş bırakın): ").strip()
                if f_cat == "": f_cat = None

                print(f"\n--- FİLTRELİ ÖNERİLER (Kullanıcı {user_id} İçin) ---")
                recs = get_recommendations(user_id, matrix, user_sim, df, method="user", filter_type=f_type,
                                           filter_category=f_cat)

                if not recs.empty:
                    print(recs.to_string(index=False))
                else:
                    print("Seçtiğiniz filtrelere uygun bir öneri bulunamadı.")
            except ValueError:
                print("Lütfen geçerli bir kullanıcı numarası girin!")

        elif secim == '4':
            print("Sistemden çıkılıyor. İyi günler!")
            break
        else:
            print("Geçersiz seçim! Lütfen 1 ile 4 arasında bir değer girin.")


if __name__ == "__main__":
    interactive_menu()