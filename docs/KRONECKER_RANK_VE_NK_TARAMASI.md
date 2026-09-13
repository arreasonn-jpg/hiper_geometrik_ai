# Faz 19/20 — Kronecker Effective Rank ve n × K Taraması

Üretim komutu:

```bash
python -m hga kronecker-rank --n-values 4,8,16 --k-values 1,2,4 \
    --out raporlar/kronecker_rank.json
```

Ham çıktı: `raporlar/kronecker_rank.json` · Kod: `hga/evaluation/kronecker_rank.py`
· Testler: `tests/test_kronecker_rank.py` (9 test)

## Neden bu deney?

README ve `hga/evaluation/sweep.py` "katman başına n² çarpan, zincir boyunca
`n^(2K)` etkileşim uzayı" diyor. Bu sayı **hesaplanıyordu ama hiç
ölçülmüyordu**. Kullanıcının "en büyük 5 eksik" listesindeki Kronecker
expressivity maddesi tam olarak buydu.

## Sonuçlar

| n | K | Eğitilebilir param | n^(2K) üst sınırı | Ölçülen rank | Maks rank | Rank kullanımı | Etkin boyut | Çöküş (aktivasyonsuz) | Kalıntı (SiLU) |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| 4 | 1 | 32 | 1.600e+01 | 16 | 16 | 1.000 | 4.45 | ÇÖKTÜ | 0.4939 |
| 4 | 2 | 64 | 2.560e+02 | 16 | 16 | 1.000 | 4.45 | ÇÖKTÜ | 0.5300 |
| 4 | 4 | 128 | 6.554e+04 | 16 | 16 | 1.000 | 4.45 | ÇÖKTÜ | 0.7237 |
| 8 | 1 | 128 | 6.400e+01 | 64 | 64 | 1.000 | 19.49 | ÇÖKTÜ | 0.5149 |
| 8 | 2 | 256 | 4.096e+03 | 64 | 64 | 1.000 | 19.49 | ÇÖKTÜ | 0.6602 |
| 8 | 4 | 512 | 1.678e+07 | 64 | 64 | 1.000 | 19.49 | ÇÖKTÜ | 0.6963 |
| 16 | 1 | 512 | 2.560e+02 | 256 | 256 | 1.000 | 98.38 | ÇÖKTÜ | 0.3776 |
| 16 | 2 | 1.024 | 6.554e+04 | 256 | 256 | 1.000 | 98.38 | ÇÖKTÜ | 0.4338 |
| 16 | 4 | 2.048 | 4.295e+09 | 256 | 256 | 1.000 | 98.38 | ÇÖKTÜ | 0.4985 |

## Bulgular

**1. Aktivasyonsuz zincir tek katmana ÇÖKÜYOR — `n^(2K)` üst sınırı boş.**

Matematiksel olarak açıktır:

    A₂(A₁ X B₁)B₂ = (A₂A₁) X (B₁B₂)

Yani K katmanlı aktivasyonsuz bir Kronecker zinciri, tek bir Kronecker
operatörüne eşittir. Bu sayısal olarak da doğrulandı: zincirin çıktısına en iyi
uyan tek doğrusal operatör aranıp kalıntı ölçüldüğünde **kalıntı ~3×10⁻⁷**
(makine hassasiyeti). Test edilen 6 çok-katmanlı konfigürasyonun **tamamı**
çöktü.

Sonuç: `n^(2K)` bu durumda erişilen bir kapasite değil, **boş bir üst
sınırdır**. K'nın ifade gücüne katkısı tam olarak sıfırdır.

**2. Derinliğin katkısı Kronecker yapısından değil, AKTİVASYONDAN geliyor.**

SiLU eklendiğinde kalıntı 0.38–0.72 aralığına çıkıyor — zincir artık tek
doğrusal operatörle temsil edilemiyor (6/6 konfigürasyonda). Aynı şey `tanh`
ve `relu` için de doğrulandı. Bu, mimarinin işe yaramadığı anlamına gelmez;
**katkının kaynağının doğru adlandırılması** gerektiği anlamına gelir.

**3. Rank özdeşliği doğrulandı — ama rank yanıltıcı bir ölçü.**

`rank(Bᵀ ⊗ A) = rank(A) · rank(B) = n · n = n²`, yani operatör **tam
rank**tır; rank kullanımı 9/9 konfigürasyonda 1.000. Buradan "kapasite tam
kullanılıyor" sonucu çıkarmak hatalı olur:

- Rank tam, ama **serbestlik derecesi yalnız 2n²**. n=16'da rank 256 iken
  eğitilebilir parametre 512, oysa aynı rank'a sahip kısıtsız bir operatör
  65.536 parametre taşır.
- **Spektral etkin boyut** (entropi tabanlı) çok daha küçük: n=16'da 256
  yerine **98.38**. Yani operatörün 256 yönü var ama enerjinin çoğu ~98 yönde
  yoğunlaşıyor. Tam rank, "tüm yönler eşit derecede kullanılıyor" demek
  değildir.

**4. Üst sınır ile öğrenilebilir kapasite arasındaki uçurum ölçekle açılıyor.**

n=16, K=4'te teorik etkileşim uzayı `4.295e+09`, gerçek eğitilebilir parametre
`2.048`. Bu, projenin başka yerlerinde de tekrarlanan ayrımın aynısıdır:
**adreslenebilir olmak ≠ öğrenilebilir olmak** (bkz. `docs/KAPASITE_CERCEVESI.md`,
C_M vs C_E/C_V ayrımı).

## Ne iddia edilmiyor

Bu modül `n^(2K)` sayısının yanlış hesaplandığını söylemiyor; **anlamının**
sınırlı olduğunu ölçüyor. Ayrıca "Kronecker yapısı işe yaramaz" da demiyor —
`docs/KRONECKER_VS_DENSE_BENCHMARK.md` eşit parametre bütçesinde Kronecker'ın
uygun görevlerde rank-1 baseline'ı yendiğini gösteriyor. Buradaki bulgu daha
dar ve daha kesin: **derinlik (K) tek başına, aktivasyon olmadan, hiçbir ifade
gücü katmaz.**
