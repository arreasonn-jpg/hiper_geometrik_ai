# HGA Kapasite Çerçevesi: P, C_I, C_M, C_E, C_V

Roadmap Faz 13 (Experience Capacity) ve Faz 14 (Verified Knowledge Capacity).

## Neden yeni iki sembol?

README bugüne kadar üç büyüklüğü dürüstçe ayırıyordu:

| Sembol | Ad | Tip |
|---|---|---|
| `P` | fiziksel/eğitilebilir parametre | ölçülen |
| `C_I` | etkileşim kapasitesi (`n^(2K)`) | üst sınır |
| `C_M` | bellek adres kapasitesi (`sözlük^pencere`) | üst sınır |

Bu doğru ama eksikti. Çünkü:

> **Adreslenebilir olmak ≠ üretilebilir olmak ≠ doğrulanabilir olmak.**

`C_M = 2.8×10⁶²` demek, sistemin 10⁶² anlamlı deneyim ürettiği anlamına
gelmez. Eksik halkalar:

| Sembol | Ad | Tanım |
|---|---|---|
| `C_E` | **Experience Capacity** | Generator'ın kısıtlar altında ürettiği ve Evaluator'ın yapısal olarak elemediği deneyim sayısı |
| `C_V` | **Verified Knowledge Capacity** | Bunlardan bağımsız verifier'ın `True`/`False` karara bağlayabildiği alt küme |

Zorunlu sıralama:

```text
C_V  ≤  C_E  ≤  C_M
```

`C_E` ve `C_V` **teorik iddia değil, ölçüm sonucudur**.

---

## Ölçüm

```bash
python -m hga kapasite --operands-max 9
```

```python
from hga.evaluation import measure_experience_capacity
from hga.experience.mini_env import AritmetikOrtam

rapor = measure_experience_capacity(
    store, verifier=AritmetikOrtam().aday_dogrula,
    relation_ids=["R_EQUALS"], sample_size=2000,
    n=256, k=4, vocab=8000, window=16,
    physical_parameters=40_524_865,
)
rapor.c_v_over_c_e      # doğrulanabilirlik oranı
rapor.decidability      # karara bağlanabilen oran (None dönenler hariç)
rapor.ordering_holds    # C_V ≤ C_E ≤ C_M
```

### `c_e_undecidable` neyi ölçer?

Verifier `None` döndüren adaylar. Bunlar **INVALID değildir** — "bilmiyorum"
bölgesidir (Faz 4 `UNCERTAIN` durumuyla aynı epistemik kategori). `C_E`'ye
girerler, `C_V`'ye girmezler. Bu ayrım olmadan "%100 doğrulama" iddiası
anlamsızdır: her şeyi reddeden bir sistem de FAR=0 üretir.

---

## Varsayılan konfigürasyonda ölçülen değerler

100-döngülük milestone koşusundan (`docs/MILESTONE_TABLOSU.md`):

| Kapasite | Sembol | Değer | Tip |
|---|---|---:|---|
| Fiziksel parametre | `P` | 40.524.865 | ölçülen |
| Etkileşim kapasitesi | `C_I` | 1.845×10¹⁹ | üst sınır |
| Bellek adres kapasitesi | `C_M` | 2.815×10⁶² | üst sınır |
| Deneyim kapasitesi | `C_E` | 1.184.832 | **ölçülen** |
| Doğrulanabilir kapasite | `C_V` | 1.180.685 | **ölçülen** |

`C_V / C_E = 0.9965` — aritmetik domaininde neredeyse her üretilen deneyim
karara bağlanabiliyor, çünkü oracle tamdır. Gerçek dilde bu oranın **çok daha
düşük** olması beklenir; kapasite çerçevesinin asıl değeri de bu farkı
ölçülebilir kılmasıdır.

`C_M` ile `C_E` arasındaki ~10⁵⁶'lık uçurum çerçevenin özetidir: sistem
10⁶² adresi adresleyebilir ama bu domainde 10⁶ deneyim üretip doğrulayabilir.

---

## Dil standardı

Bundan sonra kapasite anlatımı şu biçimde yapılır:

| ❌ Yanlış | ✅ Doğru |
|---|---|
| "10⁶² bellek" | "10⁶² kavramsal adres uzayı, 128 MB fiziksel tablo" |
| "n⁴ parametre" | "n⁴ operatör girdisi, 2n² gerçek parametre" |
| "katrilyonlarca deneyim" | "C_M = 10⁶², ölçülen C_E = 1,18×10⁶ (aritmetik domain)" |

`capacity_contract()` bu etiketleri makine-okunur biçimde de döndürür:
`c_i_is_parameter_count=False`, `c_m_is_physical_table_size=False`.
