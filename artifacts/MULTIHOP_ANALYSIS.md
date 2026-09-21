# Multi-Hop Reasoning Analizi

## Sonuclar

| Hop | 0 dist | 16 dist | 64 dist | 256 dist |
|---|---|---|---|---|
| 1 (recall) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Tum gate'ler PASS, tum metrikler 1.0000.

## Durust Uyari (Repo Kendi Diyor)

Repo docstring acikca soyluyor:
"Zincir takibi genislik-oncelikli aramadir (BFS); maliyeti kenar
sayisiyla buyur. Bu bir ogrenilmis yetenek DEGILDIR, deterministik
bir cikarimdir."

## Yorum

Bu test, HGA'nin sembolik bilgi grafigini takip edebildigini gosteriyor.
Ama bu BFS algoritmasinin sonucudur, ogrenilmis muhakeme degildir.

Gercek multi-hop reasoning icin:
1. Dil tabanli test gerekir
2. Ogrenilmis temsil ile zincir kurma
3. Belirsizlik altinda muhakeme

## Bilimsel Deger

- Pozitif: Sembolik depo + sparse memory uzerinde graf takibi saglam
- Negatif: Ogrenilmis muhakeme test edilmedi
- Durust: Repo kendi sinirini biliyor
