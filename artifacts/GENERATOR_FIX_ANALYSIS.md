# Generator Property Filter Fix

## Sorun

`ExperienceGenerator._havuz` yalnızca **tip filtresi** uyguluyordu
(`entity_type`). İlişkilerin **zorunlu özellikleri** (`requires_object_props`,
`requires_subject_props`) yok sayılıyordu.

Sonuç: consistency domain'de 1560 aday üretiliyor, 1460'ı yanlış (%93.6
halüsinasyon).

## Fix

`_havuz` genişletildi:
- Tip filtresi (mevcut)
- **Property filtresi (yeni):** her entity için `requires_*_props` kontrolü

`uret` çağrısı güncellendi:
    o_havuzu = self._havuz(varliklar, r.object_types, r.requires_object_props, store)
    s_havuzu = self._havuz(varliklar, r.subject_types, r.requires_subject_props, store)

## Test Sonuçları

Kontrollü domain (10 eligible + 10 non-eligible subj, 10 allowed + 10 blocked obj):

| Mod | Aday | Doğru | Yanlış | Halüsinasyon |
|---|---|---|---|---|
| Filtresiz | 1,560 | 100 | 1,460 | **93.6%** |
| **Filtreli** | **100** | **100** | **0** | **0.0%** |

## Etki

- 15.6× daha az aday (gereksiz üretim önlendi)
- Halüsinasyon %93.6 → %0.0
- Verifier yükü azaldı
- Yield artacak (gereksiz aday yok)

## Sınır

Bu fix **property tanımlı ilişkilerde** etkili. Aritmetik domainde property
yok, etki sıfır. Doğal dil gibi property-zengin domainlerde maksimum fayda.

## Sonuç

Generator artık **type + property aware**. Sistematik hatanın kökü kapatıldı.
