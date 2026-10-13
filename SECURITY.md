# Güvenlik Politikası

## Sorumlu Bildirim

Güvenlik açığı bildirimleri için **doğrudan bir GitHub issue açmayın**.
Bunun yerine e-posta ile ulaşın: **a.rreasonn@gmail.com**

Bildirimde bulunması gerekenler:

- Açığın kısa açıklaması ve etkilenen dosya/satır
- Yeniden üretme adımları (mümkünse minimal PoC)
- Olası etki ve önerilen düzeltme
- İletişim tercihi (anonim kalmak isteyip istemediğiniz)

Makul bir süre içinde (5 iş günü) yanıt vermeyi hedefliyoruz.

## Desteklenen Sürümler

| Sürüm | Destek |
|---|---|
| `main` (en güncel) | ✅ Aktif |
| Etiketli sürümler (`v*`) | En son 2 minor |
| Eski sürümler | ❌ |

## Bilinen Güvenlik Olayları

### SEC-2026-10-13-01 — Kör değerlendirme anahtarı public repoda

**Durum:** Kapatıldı (rotasyon bekleniyor)

**Tespit:** 13 Ekim 2026 — Birleşik denetim raporu sırasında.

**Etki:** `insan_degerlendirme_paketleri/_GIZLI_degerlendiriciye_verme/kor_anahtari.json`
dosyası, 50 prompt × 4 kol (hga/dense/transformer/symbolic) için
`item_hash → arm` eşlemesini içeriyordu. Dosya public repo'da
erişilebilir durumdaydı. Bu, değerlendirici paketlerindeki körlüğü
**teknik olarak geçersiz kılar** — paketleri indiren bir kişi hangi
metnin hangi kola ait olduğunu çapraz kontrol edebilirdi.

**Kök neden:** `.gitignore`'da `_GIZLI_*/` deseni yoktu; dosya ilk
PR (#13) sırasında yanlışlıkla eklenmiş.

**Düzeltme:**

1. Dosya ve klasör repodan kaldırıldı (bu commit).
2. `.gitignore`'a `insan_degerlendirme_paketleri/_GIZLI_*/` ve
   `kor_anahtari*.json` desenleri eklendi.
3. Kör anahtarı **rotasyona tabi tutuldu**: eski hash'ler geçersiz
   sayılıyor; değerlendirici paketleri yeni hash'ler ile yeniden
   üretilecek.
4. Eski değerlendirme sonuçları **saklanıyor** ama raporlarda
   "pre-rotation" olarak etiketlenecek.

**Not:** Git geçmişinde dosya teknik olarak hâlâ mevcuttur. Bu kayıt
şeffaflık amacıyla tutulmaktadır. İleride `git filter-repo` ile
geçmişten de temizlik yapılması planlanmaktadır.

## Tasarım İlkeleri

- **Checkpoint yükleme:** Tüm `torch.load` çağrıları `weights_only=True`
  ile yapılır. Regresyon testi: `tests/security/test_checkpoint_guvenligi.py`.
- **Kör değerlendirme:** Anahtar dosyaları repoya asla girmez. Yalnızca
  SHA-256 digest'i public olarak yayımlanır.
- **Bağımlılıklar:** Pin'lenmiş sürümler; Dependabot planlanıyor.
- **CI izinleri:** Minimum yetki; `packages: write` yalnız release
  adımında verilir.
