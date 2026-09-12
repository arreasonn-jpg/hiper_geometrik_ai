# -*- coding: utf-8 -*-
"""GeometrikTokenizer: eğitim/kodlama/kod çözme/kaydetme-yükleme testleri."""
from mimari.tokenizer import GeometrikTokenizer

METIN = (
    "Merhaba dünya! Bu bir test metnidir. Ankara Türkiye'nin başkentidir. "
    "Yapay zeka öğrenen sistemlerdir. Merhaba dünya tekrar."
)


def test_fit_sozluk_olusturur():
    tok = GeometrikTokenizer(max_vocab_size=1000).fit(METIN)
    assert tok.sozluk["<PAD>"] == 0
    assert tok.sozluk["<UNK>"] == 1
    assert tok.sozluk["<BOS>"] == 2
    assert tok.sozluk["<EOS>"] == 3
    # Tüm özel tokenlar dahil toplam kelime sayısı
    assert len(tok.sozluk) > 4
    # Fit küçük harfe çevirip sayar
    assert "merhaba" in tok.sozluk
    assert "dünya" in tok.sozluk


def test_encode_decode_roundtrip():
    tok = GeometrikTokenizer(max_vocab_size=1000).fit(METIN)
    ids = tok.encode("Merhaba dünya")
    # Sözlükteki kelimeler birebir geri çözülmeli
    assert tok.decode(ids) == "merhaba dünya"


def test_encode_bilinmeyen_kelime_unk():
    tok = GeometrikTokenizer(max_vocab_size=1000).fit(METIN)
    ids = tok.encode("kslkhyrdsyzkkelimedir")
    assert ids == [tok.UNK_ID]


def test_decode_ozel_tokenlari_atlar():
    tok = GeometrikTokenizer(max_vocab_size=1000).fit(METIN)
    assert tok.decode([0, 1, 2, 3]) == ""


def test_kaydet_yukle_roundtrip(tmp_path):
    tok = GeometrikTokenizer(max_vocab_size=1000).fit(METIN)
    yol = tmp_path / "sozluk.json"
    tok.kaydet(str(yol))

    tok2 = GeometrikTokenizer()
    tok2.yukle(str(yol))
    assert tok2.sozluk == tok.sozluk
    assert tok2.encode("Merhaba dünya") == tok.encode("Merhaba dünya")
