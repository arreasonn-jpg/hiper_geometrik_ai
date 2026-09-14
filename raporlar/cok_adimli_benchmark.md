# Çok Adımlı Çıkarım ve Uzun Bağlam (P1-004 / P1-006)

Protokol: `multi_hop_long_context_v1`  ·  veri imzası: `f8873f1d3781`
Tohumlar: [1, 2, 3]

## Doğruluk ızgarası (satır = zincir derinliği, sütun = dolgu olgu sayısı)

| hop \ dolgu | 0 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|
| 1 adım (geri çağırma) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Dejenere kontrol kolları

| kol | doğruluk | pozitif | negatif |
|---|---:|---:|---:|
| always_yes | 0.5000 | 1.0000 | 0.0000 |
| always_no | 0.5000 | 0.0000 | 1.0000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| multi_hop_inference_works | GEÇTİ |
| beats_degenerate | GEÇTİ |
| rejects_broken_chains | GEÇTİ |
| robust_to_long_context | GEÇTİ |
| memory_preserved_chain | GEÇTİ |

## Bulgular

- Çok adımlı çıkarım (hop>=2) doğruluğu 1.0000; tek adımlı geri çağırma 1.0000.
- Bağlam yükü 0 → 256 dolgu olguya çıkarıldığında doğruluk 1.0000 → 1.0000 (+0.0000).
- Tüm bağlam seviyelerinde tam doğru kalan en derin zincir: 5 adım.
- En iyi dejenere kol 0.5000; motor 1.0000 (ayrışıyor).

## Sınırlar

- Bu bir dil modeli 'context window' testi DEĞİLDİR; token penceresi, attention span veya doğal metin anlama ölçülmez. Sembolik depo + seyrek bellek üzerinde zincir takibi ölçülür.
- Zincir tek bir geçişli ilişki (öncesi) üzerinden kurulur. Bu gerçek ama DAR bir çıkarımdır; çok ilişkili karma akıl yürütme kapsam dışıdır.
- hop=1 sütunu çıkarım değil geri çağırmadır ve negatif kontrol olarak okunmalıdır; çıkarım iddiası yalnız hop>=2 için geçerlidir.
- Dolgu olgular sentetiktir ve zincirle aynı ilişkiyi kullanır; doğal metindeki anlamsal karışıklığı temsil etmez.
- Zincir takibi genişlik-öncelikli aramadır; maliyeti kenar sayısıyla büyür. Bu bir öğrenilmiş yetenek değil, deterministik bir çıkarımdır.
