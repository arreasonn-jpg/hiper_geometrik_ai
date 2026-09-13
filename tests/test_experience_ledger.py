# -*- coding: utf-8 -*-
"""Faz 24: immutable (append-only, hash-zincirli) experience ledger testleri."""
import json

from hga.experience import GENESIS_HASH, ExperienceLedger
from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru


def _aday(index: int, durum: DeneyimDurumu) -> ExperienceCandidate:
    aday = ExperienceCandidate(
        f"X-{index:03d}", "E_001", "R_001", f"E_{index:03d}",
        source=KaynakTuru.MODEL_GENERATED, state=durum,
    )
    aday.scores["novelty"] = 0.5
    if durum == DeneyimDurumu.VERIFIED:
        aday.verified_by = "deterministik-dogrulayici"
    return aday


def test_bos_defter_genesis_hash_ile_baslar():
    defter = ExperienceLedger()
    assert len(defter) == 0
    assert defter.head_hash == GENESIS_HASH
    assert defter.zincir_dogrula()


def test_reddedilen_deneyim_de_saklanir():
    defter = ExperienceLedger()
    defter.kaydet(_aday(1, DeneyimDurumu.INVALID), knowledge_version="K12",
                  reason="kural ihlali")
    defter.kaydet(_aday(2, DeneyimDurumu.UNCERTAIN), knowledge_version="K12",
                  reason="kanıt yetersiz")
    defter.kaydet(_aday(3, DeneyimDurumu.VERIFIED), knowledge_version="K12")
    ozet = defter.ozet()
    assert ozet["kayit"] == 3
    assert ozet["durumlar"] == {"INVALID": 1, "UNCERTAIN": 1, "VERIFIED": 1}
    assert ozet["bilgi_surumleri"] == ["K12"]


def test_kayitlar_zorunlu_alanlari_tasir():
    defter = ExperienceLedger()
    kayit = defter.kaydet(_aday(1, DeneyimDurumu.INVALID),
                          knowledge_version="K7", reason="çürütüldü",
                          verifier="arithmetic-env-v1", cycle=4)
    assert kayit.experience == "E_001|R_001|E_001"
    assert kayit.status == "INVALID"
    assert kayit.reason == "çürütüldü"
    assert kayit.verifier == "arithmetic-env-v1"
    assert kayit.knowledge_version == "K7"
    assert kayit.cycle == 4
    assert kayit.timestamp
    assert kayit.prev_hash == GENESIS_HASH
    assert kayit.entry_hash


def test_hash_zinciri_baglanir():
    defter = ExperienceLedger()
    a = defter.kaydet(_aday(1, DeneyimDurumu.VALID))
    b = defter.kaydet(_aday(2, DeneyimDurumu.INVALID))
    assert b.prev_hash == a.entry_hash
    assert defter.head_hash == b.entry_hash
    assert defter.zincir_dogrula()


def test_gecmise_mudahale_tespit_edilir():
    defter = ExperienceLedger()
    defter.kaydet(_aday(1, DeneyimDurumu.INVALID), reason="çürütüldü")
    defter.kaydet(_aday(2, DeneyimDurumu.VERIFIED))
    assert defter.zincir_dogrula()
    # geçmişi "düzeltmeye" çalış: INVALID → VERIFIED
    defter.entries  # kopya alır, orijinali bozmaz
    defter._entries[0].status = "VERIFIED"
    assert not defter.zincir_dogrula()


def test_entries_disari_kopya_dondurur():
    defter = ExperienceLedger()
    defter.kaydet(_aday(1, DeneyimDurumu.VALID))
    disari = defter.entries
    disari.clear()
    assert len(defter) == 1


def test_tek_deneyimin_yasam_oykusu_izlenir():
    defter = ExperienceLedger()
    aday = _aday(1, DeneyimDurumu.VALID)
    defter.kaydet(aday, reason="evaluator uyumlu buldu")
    aday.state = DeneyimDurumu.VERIFYING
    defter.kaydet(aday, reason="doğrulama başladı")
    aday.state = DeneyimDurumu.INVALID
    defter.kaydet(aday, reason="bağımsız doğrulayıcı çürüttü")
    oyku = defter.gecmis("X-001")
    assert [k.status for k in oyku] == ["VALID", "VERIFYING", "INVALID"]
    assert defter.zincir_dogrula()


def test_jsonl_kaydet_yukle_dongusu(tmp_path):
    defter = ExperienceLedger()
    for i, durum in enumerate([DeneyimDurumu.VERIFIED, DeneyimDurumu.INVALID,
                               DeneyimDurumu.CONFLICT]):
        defter.kaydet(_aday(i, durum), knowledge_version=f"K{i}")
    yol = tmp_path / "defter.jsonl"
    defter.kaydet_jsonl(str(yol))

    satirlar = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    assert len(satirlar) == 3

    geri = ExperienceLedger.yukle_jsonl(str(yol))
    assert len(geri) == 3
    assert geri.zincir_dogrula()
    assert geri.head_hash == defter.head_hash


def test_ekle_jsonl_append_only(tmp_path):
    defter = ExperienceLedger()
    yol = tmp_path / "akis.jsonl"
    for i in range(3):
        kayit = defter.kaydet(_aday(i, DeneyimDurumu.INVALID))
        defter.ekle_jsonl(str(yol), kayit)
    assert len(yol.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert ExperienceLedger.yukle_jsonl(str(yol)).zincir_dogrula()
