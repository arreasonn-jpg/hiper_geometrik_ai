# HGA için teorik çerçeve — doğrulanmış sonuçlar ve açık sınırlar

Bu belge, Kronecker zincirinin ifade gücü hakkında yalnız kanıtlanabilen
ifadeleri kullanır. `P`, `C_I^UB`, `C_M^UB`, rank, manifold boyutu ve VC
boyutu birbirinin yerine geçmez.

Makine-okunur giriş noktası:

```python
from hga.evaluation.kronecker_rank import theoretical_contract
print(theoretical_contract(n=16, k=4))
```

Ölçülen rank/çöküş protokolü:

```bash
python -m hga kronecker-rank --n-values 4,8,16 --k-values 1,2,4
```

## 1. Tek bilinear katman

Çekirdek katman `Y = A X B` ve `vec` sütun-vektörleştirmesi için:

```text
vec(Y) = (Bᵀ ⊗ A) vec(X).
```

Buradan aşağıdaki iki ifade kesindir:

1. Operatör matrisinin şekli `n² × n²`, giriş sayısı `n⁴`.
2. `rank(Bᵀ ⊗ A) = rank(A) rank(B)`.

`A` ve `B` jenerik tam rank ise operatör rank'ı `n²` olur. Bu, aynı şekilli
**her** doğrusal operatörün ifade edilebildiği anlamına gelmez. Kısıtsız bir
`n² × n²` operatör `n⁴` serbest parametre ister; tek Kronecker çarpımı çekirdeği
sadece `2n²` parametreyle taşır. Ayrıca `(A, B)` ile `(cA, B/c)` aynı operatörü
verir; sıfır olmayan jenerik Kronecker-1 manifoldunun boyutu en fazla
`2n² − 1`dir.

Dolayısıyla:

```text
operatör rank'ı = n²       ≠ serbestlik derecesi = 2n² − 1
n⁴ operatör girdisi        ≠ n⁴ eğitilebilir parametre
```

## 2. Derinlik: aktivasyonsuz çöküş teoremi

Aktivasyon, residual ve norm katmanlarını kaldırırsak iki katman için:

```text
A₂(A₁ X B₁)B₂ = (A₂A₁) X (B₁B₂).
```

İndüksiyonla K katmanın tamamı tek bir `A⋆ X B⋆` katmanına çöker. Bu durumda
`n^(2K)` ifadesi gerçekleşen fonksiyon ailesinin boyutu değildir; `K` arttıkça
lineer fonksiyon ailesi büyümez. Bu sonuç `KureselZincir`in aktivasyonsuz
sürümü için cebirsel bir eşitliktir ve `kronecker-rank` testi tarafından
sayısal kalıntı ile de doğrulanır.

Mevcut modelde SiLU, LayerNorm ve residual bağlantı vardır. SiLU ile zincir
tek doğrusal Kronecker operatörüne indirgenemez; ancak bu gözlem **evrensel
yaklaştırma**, VC boyutu veya veri üzerinde avantaj kanıtı değildir. Derinliğin
katkısı bu durumda Kronecker çarpımından tek başına değil, doğrusal olmayan
kompozisyondan gelir.

## 3. Etkin rank / etkin boyut

Matrisin sayısal rank'ı yalnız sıfır olmayan tekil değerleri sayar. Bu nedenle
rank tam olsa bile bilgi birkaç yönde yoğunlaşabilir. Protokol iki ek ölçü
raporlar:

- **participation ratio:** `(Σ σᵢ)² / Σ σᵢ²`;
- **spectral entropy effective dimension:** `exp(-Σ pᵢ log pᵢ)`,
  `pᵢ = σᵢ² / Σ σⱼ²`.

Bunlar veri, seed, eğitim ve toleransa bağımlı ölçümlerdir; teorik alt/üst
sınır değildir. Yine de `rank = n²` sonucunun “bütün yönler eşit kullanıldı”
şeklinde yanlış yorumlanmasını önler.

## 4. VC boyutu: ne biliniyor, ne bilinmiyor

Bu depoda **tam trainable HGA zinciri için kapalı-form bir VC veya
pseudo-dimension teoremi kurulmuş değildir**. Bunun nedeni salt parametre
sayısı değildir: SiLU analitik/doğrusal olmayan bir aktivasyondur, LayerNorm
örnek-bağımlı normalizasyon yapar ve ağırlıklar Kronecker biçiminde paylaşılır.
`W` parametre sayısını doğrudan “VC = W” saymak geçerli değildir.

Kesin ve daha dar ifade şudur:

> Bilinear gövde sabitlenir ve yalnız `d=n²` boyutlu özellik üzerindeki affine
> ikili readout eğitilirse, classifier VC boyutu en fazla `d+1`dir (genel
> konumdaki uygun bir domain üzerinde eşitlik sağlanabilir).

Bu değer **tam zincire taşınamaz**. `theoretical_contract()` bunu
`fixed_feature_affine_readout_vc_dimension_upper_bound` olarak verir ve tam
zincir alanını özellikle `null` / `NOT_ESTABLISHED` bırakır.

Parçalı-polynomial aktivasyonlarla tanımlanan ağlar için literatürde parametre
sayısı ve katman sayısına bağlı genel VC/pseudo-dimension üst sınırları vardır;
SiLU + LayerNorm mimarisine ek koşullar olmadan bu sınır burada alıntı yoluyla
uygulanmaz. Böyle bir sonuç için mimariyi açıkça uygun fonksiyon sınıfına
indirgemek veya ayrı bir kanıt sunmak gerekir.

### Ampirik shattering protokolü

Bir sonraki teorik/deneysel çalışma aşağıdaki sayıyı en fazla **alt sınır
kanıtı** olarak raporlayabilir:

1. Girdileri ve rastgele label dizilerini hash ile sabitle.
2. Her `m` için birden çok labelleme ve seed üzerinde tam eğitim yap.
3. Tüm labellemeler sıfır eğitim hatası verirse sadece `VC ≥ m` de.
4. Başarısızlık `VC < m` kanıtı değildir; optimizasyon başarısız olmuş olabilir.
5. Benchmark test skorunu VC ile karıştırma.

## 5. Yayın dili sözleşmesi

İzinli ifadeler:

- “Tek bilinear katmanın `n² × n²` Kronecker yapılandırılmış operatörü vardır.”
- “Aktivasyonsuz K katman lineer olarak tek katmana çöker.”
- “Etkin rank ve spektral boyut ölçüldü; rapor ve seed belirtilmiştir.”
- “Sabit temsilde affine readout için VC üst sınırı `n²+1`dir.”

İzinli olmayan ifadeler:

- “`n^(2K)` gerçek kapasite/parametre/VC boyutudur.”
- “Tam HGA zincirinin VC boyutu `P`, `n⁴` veya `n^(2K)`dir.”
- “Tam rank, kısıtsız dense operatör veya üstün genelleme demektir.”

Bu belge CKPT-000'un teori açığını görünmez kılmaz: açık problem artık
ölçülebilir ve yanlış iddiadan ayrılmıştır. Hakemli bir teorem için ayrı ispat,
varsayımlar ve dış değerlendirme gerekir.
