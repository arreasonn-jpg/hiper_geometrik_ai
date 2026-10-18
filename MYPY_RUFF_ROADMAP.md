# mypy & ruff Kademeli Sıkılaştırma Yol Haritası

**Hazırlayan:** P0-10 (birleşik audit raporu)
**Tarih:** 2026-10-13
**Durum:** Faz 0 tamamlandı, Faz 1+ planlandı

## Mevcut Durum

| Araç | Kural Seti | Hata | Yorum |
|---|---|---|---|
| ruff | `E, F, W, I` | 0 | Temel set — CI yeşil |
| ruff | `+SIM` | 25 | Bu turda **açılmadı** |
| ruff | `+B` | 112 | 92'si B905 (toplu değişiklik gerektirir) |
| ruff | `+S` | 3272 | 3201'i S101 (test assert'leri) |
| ruff | `+UP` | 3290 | 2183'ü UP006 (typing.List → list) |
| mypy | Mevcut config | 0 | Python 3.11'e yükseltildi |
| mypy | `check_untyped_defs` | ? | Python sürüm düzeltmesi sonrası ölçülecek |

## Faz 0 — Tamamlananlar (2026-10-13)

- [x] Ruff I001 düzeltmesi (`tests/security/test_checkpoint_guvenligi.py`)
- [x] `pyproject.toml`: mypy `python_version = "3.11"` (numpy 2.5 stub uyumu)
- [x] `tests/**` için `B011` per-file-ignore (doctest assert'leri)

## Faz 1 — Kademeli Sıkılaştırma (2-4 hafta)

### Ruff: SIM (simplify) — 25 hata
**Öncelik:** 🟢 Düşük risk, kazanç yüksek
**Eylem:**
1. `ruff check --select SIM --fix .` (5 otomatik düzeltme)
2. Kalan 20 manuel inceleme
3. `pyproject.toml`'a `SIM` ekle

**Kurallar:**
- SIM102: collapsible-if (6)
- SIM108: if-else → ternary (5)
- SIM105: `contextlib.suppress` (5)
- SIM300: yoda conditions (4)
- Diğer tekil durumlar (5)

### Ruff: B (bugbear) — 112 hata
**Öncelik:** 🟠 Orta risk, çok sayıda toplu değişiklik
**Eylem:**
- B905 (`zip(strict=)`): 92 yer — **toplu mekanik düzeltme**, ama her `zip`'in gerçekten aynı uzunlukta olduğunu doğrulamak gerekir. Yanlış `strict=True` çalışma zamanı hatası verir. **Dikkatli yapılmalı.**
- B007: unused-loop-control-variable (7)
- B006: mutable-argument-default (5) — 🔴 **potansiyel bug**, öncelikli
- B023, B020, B017, B018: tek tük

**Öneri:** B006 öncelikli düzelt, B905'i tek bir sprint'te ele al, geri kalanı ignore et.

### mypy: check_untyped_defs
**Öncelik:** 🟠 Orta
**Eylem:**
1. Python sürüm düzeltmesi sonrası ölç
2. `hga/experiments/` (çekirdek) için aç, kalanı kademeli
3. Hedef: Faz 1 sonunda `hga/` tamamı

### mypy: disallow_untyped_defs
**Öncelik:** 🔴 Yüksek efor
**Eylem:** Faz 2'ye ertele (bkz. aşağıda)

## Faz 2 — Kapsamlı Modernizasyon (1-3 ay)

### Ruff: UP (pyupgrade) — 3290 hata
**Neden ertelendi:** UP006 (2183 hata) `typing.List` → `list` gibi mekanik ama **çok sayıda dosyayı** etkiler. Ayrı bir sprint + dikkatli review gerektirir.

**Eylem sırası:**
1. UP009 (233, `# -*- coding: utf-8 -*-` gereksiz) — kolay
2. UP015 (31, `open(f, "r")` → `open(f)`) — kolay
3. UP037, UP007, UP012, UP034 — kolay
4. UP006 (2183) — toplu, ayrı PR
5. UP045 (448) — toplu, ayrı PR

### Ruff: S (bandit) — 3272 hata
**Neden ertelendi:** S101 (`assert`) — 3201 hata. Test'lerde assert normal, üretimde `python -O` ile silinir. Çözüm: `tests/**` için S101 ignore, üretim için `if not x: raise` yapısı.

**Eylem sırası:**
1. `pyproject.toml`'a S101 sadece `tests/**` için ignore
2. S311 (24, `random`), S603/S607 (17+3, `subprocess`), S105 (16, hardcoded password) incele
3. S110, S112, S104, S108 tek tük

**Not:** S105'lerin çoğu muhtemelen test/mock sırları — false positive.

### mypy: disallow_untyped_defs
**Eylem:** Her modülü tek tek ele al. Hedef: `hga/` tamamı, sonra `mimari/` ve `egitim/`.

## Faz 3 — Sürekli Disiplin

- Dependabot + pip-audit (bağımlılık taraması)
- CodeQL veya Semgrep (kod güvenlik taraması)
- Aylık ruff/mypy sürüm güncellemesi + kural seti gözden geçirme
- Yeni kod eklerken CI'da strict kurallar (önce warning → sonra error)

## Kabul Kriterleri (10/10 için)

- [ ] `ruff check --select E,F,W,I,B,SIM` → 0 hata
- [ ] `ruff check --select S` → sadece test dosyalarında
- [ ] `mypy --check-untyped-defs hga/` → 0 hata
- [ ] `mypy --disallow-untyped-defs hga/` → 0 hata
- [ ] `pytest --cov=hga --cov-fail-under=85` → PASS
- [ ] CI süresi < 15 dk

## Referanslar

- P0-10 (birleşik audit raporu)
- Commit `f73723c` (Kronecker rank güvenlik kapısı)
