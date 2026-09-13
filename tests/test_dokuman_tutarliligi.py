# -*- coding: utf-8 -*-
"""Belgelerdeki sayısal iddialar koda karşı doğrulanır.

Belgeler kolayca eskir: kod değişir, README'deki tablo eski sayıyı gösterir
ve kimse fark etmez. Bu testler README/docs içindeki ölçüm iddialarını
canlı çıktıyla karşılaştırır; sayı değişirse test kırılır ve belgenin
güncellenmesi gerektiği anlaşılır.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

KOK = pathlib.Path(__file__).resolve().parents[1]
README = (KOK / "README.md").read_text(encoding="utf-8")
DOC_EPISTEMIK = (KOK / "docs" / "EPISTEMIK_BENCHMARK.md").read_text(encoding="utf-8")
DOC_VERIM = (KOK / "docs" / "VERIM_METRIKLERI.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Referans bütünlüğü
# --------------------------------------------------------------------------

def test_readme_docs_referanslari_mevcut():
    eksik = [yol for yol in sorted(set(re.findall(r"docs/[A-Z_0-9]+\.md", README)))
             if not (KOK / yol).exists()]
    assert not eksik, f"README var olmayan belgeye atıf yapıyor: {eksik}"


def test_readme_cli_komutlari_gercekten_tanimli():
    ana = (KOK / "hga" / "__main__.py").read_text(encoding="utf-8")
    blok = ana.split('p.add_argument("komut"')[1].split("])")[0]
    tanimli = set(re.findall(r'"([a-z0-9-]+)"', blok))
    kullanilan = set(re.findall(r"python -m hga ([a-z0-9-]+)", README))
    bilinmeyen = kullanilan - tanimli
    assert not bilinmeyen, f"README tanımsız CLI komutu gösteriyor: {sorted(bilinmeyen)}"


def test_her_cli_komutu_ci_smoke_testinde_kosuluyor():
    """Tanımlı her komut CI'da en az bir kez çalıştırılmalı.

    Koşulmayan komut sessizce bozulur: 7 komut (graf, kesif, dogrulama,
    halusinasyon, sweep, tokenizer, benchmark-rapor) uzun süre CI kapsamı
    dışındaydı. Bu test kapsamı kalıcı kılar.
    """
    ana = (KOK / "hga" / "__main__.py").read_text(encoding="utf-8")
    blok = ana.split('p.add_argument("komut"')[1].split("])")[0]
    tanimli = set(re.findall(r'"([a-z0-9-]+)"', blok))

    ci = (KOK / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    kosulan = set(re.findall(r"python -m hga ([a-z0-9-]+)", ci))

    # Dosya yolu/argüman gerektiren, smoke'a uygun olmayan komutlar.
    muaf = {"ozet", "manifest", "veri-kalite", "perplexity", "checkpoint-rapor",
            "gercek-veri", "veri-canli-smoke", "observability"}

    kapsanmayan = tanimli - kosulan - muaf
    assert not kapsanmayan, (
        "Bu CLI komutları CI smoke testinde hiç çalıştırılmıyor: "
        f"{sorted(kapsanmayan)}. Ya CI'a ekleyin ya muaf listesine."
    )


@pytest.mark.parametrize("yol", [
    "raporlar/epistemik_benchmark.json",
    "raporlar/epistemik_benchmark.md",
    "raporlar/verim_metrikleri.json",
    "raporlar/verim_metrikleri.md",
])
def test_kuratorlenen_raporlar_mevcut(yol):
    assert (KOK / yol).exists(), f"Küratörlenen rapor eksik: {yol}"


def test_cok_tohumlu_verim_raporu_tek_tohum_siniri_iddia_etmez():
    """Çok tohumlu rapor "güven aralığı için çok tohum gerekir" DİYEMEZ.

    Bu cümle tek koşu için doğru ama toplu raporda kendi GA tablosuyla
    çelişiyordu. Regresyon kapısı.
    """
    md = (KOK / "raporlar" / "verim_metrikleri.md").read_text(encoding="utf-8")
    assert "%95 GA" in md, "Toplu raporda güven aralığı tablosu olmalı"
    assert "tek tohumun tek koşusudur" not in md, (
        "Çok tohumlu rapor hâlâ tek-tohum sınırını iddia ediyor; "
        "kendi GA tablosuyla çelişiyor."
    )


def test_verim_raporu_bulgulari_hangi_tohuma_ait_oldugunu_soyluyor():
    """Bulgular tohum 1'e ait; tablo 5 tohumun ortalaması. Etiket şart."""
    md = (KOK / "raporlar" / "verim_metrikleri.md").read_text(encoding="utf-8")
    assert re.search(r"##\s*Bulgular\s*\(tohum\s*\d+\)", md), (
        "Bulgular başlığı hangi tohuma ait olduğunu belirtmeli; aksi halde "
        "okur bunları tablodaki ortalamayla karıştırır."
    )


def test_verim_json_toplu_istatistik_iceriyor():
    """Markdown'daki GA'lar makine-okunur da olmalı."""
    veri = json.loads((KOK / "raporlar" / "verim_metrikleri.json").read_text(encoding="utf-8"))
    assert "aggregate" in veri, "JSON toplu istatistik (aggregate) içermeli"
    for alan in ("experience_yield", "novelty_yield", "useful_experience_yield",
                 "generalization_yield", "verified_information_density"):
        ozet = veri["aggregate"][alan]
        assert ozet["ci_lower"] <= ozet["mean"] <= ozet["ci_upper"], (
            f"{alan}: ortalama kendi güven aralığının dışında"
        )


# --------------------------------------------------------------------------
# Epistemik benchmark iddiaları
# --------------------------------------------------------------------------

def test_epistemik_fixture_boyutu_belgeyle_uyumlu():
    from hga.evaluation.epistemic import EpistemicDataset

    veri = EpistemicDataset()
    assert len(veri.cases) == 28, "Vaka sayısı değişti; belgeler güncellenmeli"
    assert len(veri.entities) == 16
    assert len(veri.relations) == 2
    assert "28 vaka" in DOC_EPISTEMIK


def test_epistemik_negatif_kontrol_tablosu_belgeyle_uyumlu():
    """README ve belgedeki dejenere kol tablosu canlı çıktıyla aynı olmalı."""
    from hga.evaluation.epistemic import run_epistemic_baselines

    beklenen = {
        "always_valid": (0.286, 1.0, 0.0, 1.0),
        "always_abstain": (0.321, 0.0, 1.0, 0.0),
        "always_invalid": (0.286, 0.0, 0.0, 1.0),
    }
    for kol in run_epistemic_baselines():
        acc, known, unknown, fc = beklenen[kol.arm]
        assert round(kol.accuracy, 3) == acc, kol.arm
        assert kol.known_accuracy == known, kol.arm
        assert kol.unknown_accuracy == unknown, kol.arm
        assert kol.false_confidence_rate == fc, kol.arm


def test_epistemik_ana_metrikler_belgedeki_degerlerde():
    from hga.evaluation.epistemic import run_epistemic_benchmark

    rapor = run_epistemic_benchmark()
    assert rapor.false_confidence_rate == 0.0
    assert rapor.unknown_accuracy == 1.0
    assert rapor.accuracy == 1.0
    # Ayrım sebep alanıyla yapılabiliyor; durum kodu seviyesinde hâlâ tek durum.
    assert rapor.epistemic_resolution["distinguishable"] is True
    assert rapor.epistemic_resolution["distinguishable_by_state"] is False


def test_belge_unknown_uncertain_ayrimini_dogru_anlatiyor():
    """Sınır kapatıldı; belgeler hem eski ölçümü hem çözümü anlatmalı.

    Önemli olan "başardık" demek değil, iki çözünürlük seviyesinin ayrı ayrı
    raporlandığının belgede görünmesi: durum kodu hâlâ tek (`by_state=false`),
    ayrım sebep alanında (`by_reason=true`).
    """
    for metin, ad in ((DOC_EPISTEMIK, "docs"), (README, "README")):
        assert "KAYIT_YOK" in metin, ad
        assert "OZELLIK_YOK" in metin, ad
        assert "distinguishable_by_state" in metin, ad
        assert "distinguishable_by_reason" in metin, ad
    assert "mimari sınır" in DOC_EPISTEMIK


def test_belirsizlik_sebebi_enum_belgeyle_uyumlu():
    """Belgede listelenen sebep üyeleri gerçekten enum'da olmalı."""
    from hga.knowledge import BelirsizlikSebebi

    uyeler = {m.value for m in BelirsizlikSebebi}
    assert uyeler == {"YOK", "KAYIT_YOK", "OZELLIK_YOK", "DOGRULAYICI_KARARSIZ"}
    for uye in uyeler:
        assert uye in DOC_EPISTEMIK, f"Belgede eksik sebep üyesi: {uye}"


# --------------------------------------------------------------------------
# Verim metrikleri iddiaları
# --------------------------------------------------------------------------

def test_verim_gy_tablosu_belgeyle_uyumlu():
    """docs/VERIM_METRIKLERI.md içindeki GY monotonluk tablosu."""
    from hga.experience.verim import run_yield_experiment

    beklenen = {10: (14, 0.1628), 20: (31, 0.3605), 30: (52, 0.6047),
                40: (56, 0.6512)}
    for dongu, (karar, gy) in beklenen.items():
        rapor = run_yield_experiment(
            cycles=dongu, batch_size=32, initial_facts=40,
            operands_max=15, negatives_per_fact=3, seed=1,
        )
        assert rapor.holdout_after["decided"] == karar, f"cycles={dongu}"
        assert round(rapor.generalization_yield, 4) == gy, f"cycles={dongu}"
        # Karar verilen her örnekte isabet tam olmalı.
        assert rapor.holdout_after["accuracy_on_decided"] == 1.0


def test_verim_ham_sayimlari_belgeyle_uyumlu():
    from hga.experience.verim import run_yield_experiment

    rapor = run_yield_experiment(cycles=30, batch_size=32, initial_facts=40,
                                 operands_max=15, negatives_per_fact=3, seed=1)
    assert rapor.generated == 960
    assert rapor.verified == 220
    assert rapor.distinct_new_facts == 178
    assert rapor.useful_facts == 148
    assert rapor.duplicate_generations == 224
    assert rapor.memory_collisions == 72
    assert rapor.incorrect_facts == 0


def test_verim_ayrisma_yonu_belgedeki_iddiayi_dogruluyor():
    """Belge 'EY > NY > UEY' diyor; sıralama gerçekten bu olmalı."""
    from hga.evaluation.statistics import summarize_seed_metric
    from hga.experience.verim import run_yield_experiment

    raporlar = [
        run_yield_experiment(cycles=30, batch_size=32, initial_facts=40,
                             operands_max=15, negatives_per_fact=3, seed=s)
        for s in (1, 2, 3, 4, 5)
    ]
    ey = summarize_seed_metric([r.experience_yield for r in raporlar])["mean"]
    ny = summarize_seed_metric([r.novelty_yield for r in raporlar])["mean"]
    uey = summarize_seed_metric([r.useful_experience_yield for r in raporlar])["mean"]
    assert ey > ny > uey, f"Ayrışma yönü değişti: EY={ey} NY={ny} UEY={uey}"
    assert round(ey, 4) == 0.2385
    assert round(ny, 4) == 0.1904
    assert round(uey, 4) == 0.1565
    assert "EY > NY > UEY" in DOC_VERIM


def test_belge_gy_olu_metrik_hikayesini_koruyor():
    """İki regresyonun nasıl yakalandığı belgede kalmalı (kurumsal hafıza)."""
    for anahtar in ("ölü metrik", "fonksiyonel teklik", "0.6375"):
        assert anahtar in DOC_VERIM, f"Belgeden kayboldu: {anahtar}"


def test_belge_istatistiksel_guc_uyarisini_iceriyor():
    """5 seedde p<0.05 imkânsızlığı verim belgesinde yazmalı."""
    assert "p<0.05" in DOC_VERIM
    assert "0.0625" in DOC_VERIM


# --------------------------------------------------------------------------
# Belgelerdeki TABLOLAR gerçekten ayrıştırılıp karşılaştırılır
#
# Yalnız canlı değeri doğrulamak yetmez: belgedeki tablo eskirse kimse fark
# etmez. Aşağıdaki testler Markdown tablolarını okuyup sayıyı karşılaştırır.
# --------------------------------------------------------------------------

def _tablo_degerleri(metin: str, etiket: str) -> list:
    """`| etiket | 0.123 | ... |` satırındaki ondalık sayıları döndür."""
    for satir in metin.splitlines():
        if satir.strip().startswith("|") and etiket in satir:
            return [float(x) for x in re.findall(r"\d+\.\d+", satir)]
    raise AssertionError(f"Tablo satırı bulunamadı: {etiket!r}")


@pytest.mark.parametrize("etiket,alan", [
    ("EY (klasik)", "experience_yield"),
    ("yenilik", "novelty_yield"),
    ("kullanışlı", "useful_experience_yield"),
    ("genelleme", "generalization_yield"),
    ("bit/deneyim", "verified_information_density"),
])
def test_readme_verim_tablosu_canli_degerle_ayni(etiket, alan):
    """README'deki 5 tohumlu verim tablosu eskirse bu test kırılır."""
    from hga.evaluation.statistics import summarize_seed_metric
    from hga.experience.verim import run_yield_experiment

    raporlar = [
        run_yield_experiment(cycles=30, batch_size=32, initial_facts=40,
                             operands_max=15, negatives_per_fact=3, seed=s)
        for s in (1, 2, 3, 4, 5)
    ]
    ozet = summarize_seed_metric([getattr(r, alan) for r in raporlar])
    yazili = _tablo_degerleri(README, etiket)
    assert yazili[0] == round(ozet["mean"], 4), (
        f"README'deki '{etiket}' ortalaması eskimiş: "
        f"yazılı={yazili[0]} canlı={round(ozet['mean'], 4)}"
    )
    # Tabloda ayrıca %95 GA alt/üst sınırı var.
    assert yazili[1] == round(ozet["ci_lower"], 4), etiket
    assert yazili[2] == round(ozet["ci_upper"], 4), etiket


@pytest.mark.parametrize("kaynak_adi", ["README", "DOC_EPISTEMIK"])
@pytest.mark.parametrize("kol_adi", ["always_valid", "always_abstain", "always_invalid"])
def test_negatif_kontrol_tablolari_canli_degerle_ayni(kaynak_adi, kol_adi):
    """Hem README hem belge içindeki dejenere kol tablosu doğrulanır."""
    from hga.evaluation.epistemic import run_epistemic_baselines

    metin = README if kaynak_adi == "README" else DOC_EPISTEMIK
    kol = next(a for a in run_epistemic_baselines() if a.arm == kol_adi)
    yazili = _tablo_degerleri(metin, f"`{kol_adi}`")
    beklenen = [round(kol.accuracy, 3), kol.known_accuracy,
                kol.unknown_accuracy, kol.false_confidence_rate]
    assert yazili == pytest.approx(beklenen, abs=1e-9), (
        f"{kaynak_adi} içindeki {kol_adi} satırı eskimiş: "
        f"yazılı={yazili} canlı={beklenen}"
    )


@pytest.mark.parametrize("dongu,karar,gy", [
    (10, 14, 0.1628), (20, 31, 0.3605), (30, 52, 0.6047), (40, 56, 0.6512),
])
def test_belgedeki_gy_tablosu_satir_satir_ayristirilir(dongu, karar, gy):
    """docs/VERIM_METRIKLERI.md GY tablosu satırı canlı çıktıyla aynı olmalı."""
    from hga.experience.verim import run_yield_experiment

    rapor = run_yield_experiment(cycles=dongu, batch_size=32, initial_facts=40,
                                 operands_max=15, negatives_per_fact=3, seed=1)
    satir = next(
        s for s in DOC_VERIM.splitlines()
        if s.strip().startswith(f"| {dongu} |")
    )
    assert f"{karar}/86" in satir, f"Belgedeki karar sayısı eskimiş: {satir}"
    assert f"{gy:.4f}" in satir, f"Belgedeki GY eskimiş: {satir}"
    assert rapor.holdout_after["decided"] == karar
    assert round(rapor.generalization_yield, 4) == gy
