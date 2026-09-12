# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Subword Reconstructor BPE Tokenizer
=========================================================

Güvenlik/notlar:
  * Türkçe büyük/küçük harf dönüşümü Python'un varsayılan ``str.lower``
    davranışına bırakılmaz. Özellikle ``İ`` harfi ``i`` + birleştirici nokta
    üretmemelidir; ``I`` ise Türkçe ``ı`` olarak küçültülür.
  * Varsayılan sözlük bütçesi yeterliyse 256 adet byte-fallback token'ı
    (``<0x00>`` ... ``<0xFF>``) rezerve edilir. Böylece eğitim korpusunda
    görülmeyen Unicode karakterleri ``<UNK>``'a düşmeden geri çözülebilir.
  * ``ozel_tokenlari_dogrula`` ve ``vocab_tutarliligi`` eğitimden önce sessiz
    embedding/tokenizer uyumsuzluklarını yakalamak için kullanılabilir.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence

try:  # Tokenizer'ın sözlük/encode kısmı torch olmadan da kullanılabilsin.
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch kurulu olmayan hafif test ortamı
    torch = None  # type: ignore


_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class BPETokenizer:
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    OZEL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(256)]
    END_WORD = "</w>"

    def __init__(self, baglam_penceresi: int = 16, max_vocab_size: int = 8000,
                 min_freq: int = 2):
        self.baglam_penceresi = int(baglam_penceresi)
        self.max_vocab_size = int(max_vocab_size)
        self.min_freq = int(min_freq)
        self.kelime_to_id: Dict[str, int] = {}
        self.id_to_kelime: Dict[int, str] = {}
        self.subword_set: set = set()
        self.sozluk_boyutu: int = 0
        self._egitildi = False
        self._cache: Dict[str, List[str]] = {}

    # ── Normalizasyon ───────────────────────────────────────────────────
    @staticmethod
    def _turkce_kucult(metin: str) -> str:
        """Türkçe karakterleri koruyan deterministik küçültme.

        Python ``"İ".lower()`` sonucunda ``"i\u0307"`` ürettiği için BPE
        sözlüğünde görünmez bir birleştirici nokta oluşabiliyordu. Bu yöntem
        ``İ → i`` ve ``I → ı`` eşlemesini önce yapar, sonra NFC normalize eder.
        """
        metin = str(metin).replace("İ", "i").replace("I", "ı")
        return unicodedata.normalize("NFC", metin.lower())

    def _metni_temizle(self, metin: str) -> List[str]:
        if not metin:
            return []
        metin = self._turkce_kucult(metin)
        metin = re.sub(r"([.,!?;:\"'()\[\]{}…])", r" \1 ", metin)
        metin = re.sub(r"\s+", " ", metin).strip()
        return [k for k in metin.split(" ") if k]

    # ── Sözlük eğitimi ──────────────────────────────────────────────────
    def fit_on_text(self, metin: str, verbose: bool = True) -> "BPETokenizer":
        t0 = time.time()
        kelimeler = self._metni_temizle(metin) or ["merhaba", "dünya"]
        wfreq = Counter(kelimeler)
        top_words = dict(wfreq.most_common(12000))

        k_sembolleri = {tuple(list(w) + [self.END_WORD]): f
                        for w, f in top_words.items()}
        char_set = {s for syms in k_sembolleri for s in syms}

        merges = []
        for _step in range(5000):
            ciftler = Counter()
            for s, f in k_sembolleri.items():
                if len(s) < 2:
                    continue
                for i in range(len(s) - 1):
                    ciftler[(s[i], s[i + 1])] += f
            if not ciftler:
                break
            en_iyi, en_skor = ciftler.most_common(1)[0]
            if en_skor < self.min_freq:
                break

            a, b = en_iyi
            birlesik = a + b
            yeni = {}
            for s, f in k_sembolleri.items():
                s_list = list(s)
                i = 0
                out = []
                while i < len(s_list):
                    if i < len(s_list) - 1 and s_list[i] == a and s_list[i + 1] == b:
                        out.append(birlesik)
                        i += 2
                    else:
                        out.append(s_list[i])
                        i += 1
                yeni[tuple(out)] = yeni.get(tuple(out), 0) + f
            k_sembolleri = yeni
            merges.append(birlesik)

        self.kelime_to_id = {t: i for i, t in enumerate(self.OZEL)}
        next_id = len(self.OZEL)

        # Kelime sonu belirteci decode sırasında boşluk üretir; byte-only
        # fallback modunda bile kelime sınırını kaybetmemek için önceliklidir.
        if next_id < self.max_vocab_size:
            self.kelime_to_id[self.END_WORD] = next_id
            next_id += 1

        # Byte-level fallback: varsayılan 8000'lik sözlükte maliyeti küçüktür
        # ve bilinmeyen Unicode karakterlerini <UNK>'a düşmekten kurtarır.
        # Çok küçük deney sözlüklerinde BPE karakter/merge kapasitesini tamamen
        # boğmamak için yalnız tüm 256 byte sığabiliyorsa açılır.
        if self.max_vocab_size - next_id >= 256:
            for tok in self.BYTE_TOKENS:
                self.kelime_to_id[tok] = next_id
                next_id += 1

        # Önce karakterler (Türkçe harfler asla düşmesin), sonra merge'ler.
        sirali_tokenler = sorted(char_set) + sorted(set(merges))
        for tok in sirali_tokenler:
            if next_id >= self.max_vocab_size:
                break
            if tok not in self.kelime_to_id:
                self.kelime_to_id[tok] = next_id
                next_id += 1

        self.id_to_kelime = {v: k for k, v in self.kelime_to_id.items()}
        self.sozluk_boyutu = len(self.kelime_to_id)
        self.subword_set = set(self.kelime_to_id.keys())
        self._cache.clear()
        self._egitildi = True

        # Eğitim sonrası tutarlılık hataları sessiz kalmasın.
        self.vocab_tutarliligi(strict=True)

        if verbose:
            print(f"[BPE PERFECT] ✅ Vocab ({self.sozluk_boyutu}) "
                  f"{time.time() - t0:.2f} saniyede oluşturuldu!")
        return self

    # ── Parçalama / byte fallback ───────────────────────────────────────
    @property
    def byte_fallback_var(self) -> bool:
        return all(tok in self.kelime_to_id for tok in self.BYTE_TOKENS)

    def _byte_tokenlari(self, metin: str) -> List[str]:
        if not self.byte_fallback_var:
            return [metin] if metin in self.subword_set else [self.OZEL[self.UNK_ID]]
        return [f"<0x{b:02X}>" for b in metin.encode("utf-8")]

    def _fast_parcala(self, kelime: str) -> List[str]:
        kelime = self._turkce_kucult(kelime)
        if kelime in self._cache:
            return self._cache[kelime]

        # Noktalama veya emoji gibi tekil/alfasayısal olmayan birimler.
        if len(kelime) == 1 or not kelime.isalnum():
            if kelime in self.subword_set:
                res = [kelime]
            else:
                res = self._byte_tokenlari(kelime)
            self._cache[kelime] = res
            return res

        res: List[str] = []
        w = kelime + self.END_WORD
        i = 0
        n = len(w)

        while i < n:
            matched = False
            for l in range(min(n - i, 24), 0, -1):
                sub = w[i: i + l]
                if sub in self.subword_set:
                    res.append(sub)
                    i += l
                    matched = True
                    break
            if matched:
                continue

            # Kelime sonu belirteci sözlükte yoksa literal '</w>' parçalarını
            # byte fallback'e düşürüp çıktı metnine sızdırma.
            if w.startswith(self.END_WORD, i):
                if self.END_WORD in self.subword_set:
                    res.append(self.END_WORD)
                break

            res.extend(self._byte_tokenlari(w[i]))
            i += 1

        if len(self._cache) < 100000:
            self._cache[kelime] = res
        return res

    def egitim_token_listesi(self, metin: str) -> List[str]:
        kelimeler = self._metni_temizle(metin)
        return [p for w in kelimeler for p in self._fast_parcala(w)]

    def _ids_from_text(self, metin, pad: bool = False) -> List[int]:
        klm = self._metni_temizle(metin) if isinstance(metin, str) else list(metin)
        pieces = [p for w in klm for p in self._fast_parcala(w)]
        ids = [self.kelime_to_id.get(p, self.UNK_ID) for p in pieces]
        if pad:
            if len(ids) >= self.baglam_penceresi:
                ids = ids[-self.baglam_penceresi:]
            else:
                ids = [self.PAD_ID] * (self.baglam_penceresi - len(ids)) + ids
        # Geriye dönük uyumluluk: boş metin en az bir PAD döndürür.
        return ids or [self.PAD_ID]

    def text_to_ids(self, metin, pad: bool = True):
        ids = self._ids_from_text(metin, pad=pad)
        if torch is None:  # pragma: no cover - yalnız torch'suz ortamda
            raise ImportError("text_to_ids torch.Tensor döndürür; torch kurulu değil. "
                              "Torch'suz kullanım için encode(...) çağırın.")
        return torch.tensor(ids or [self.PAD_ID], dtype=torch.long)

    # ── Decode ──────────────────────────────────────────────────────────
    @staticmethod
    def _byte_degeri(tok: str) -> Optional[int]:
        m = _BYTE_RE.match(tok)
        return int(m.group(1), 16) if m else None

    @classmethod
    def _normal_fragment(cls, tok: str) -> str:
        if tok in cls.OZEL:
            return ""
        if tok == cls.END_WORD:
            return " "
        return tok.replace(cls.END_WORD, " ") if tok.endswith(cls.END_WORD) else tok

    def id_to_text(self, idx: int) -> str:
        tok = self.id_to_kelime.get(int(idx), "")
        b = self._byte_degeri(tok)
        if b is not None:
            return bytes([b]).decode("utf-8", errors="replace")
        return self._normal_fragment(tok)

    def decode(self, ids: Sequence[int]) -> str:
        parcalar: List[str] = []
        byte_buf = bytearray()

        def flush_bytes():
            nonlocal byte_buf
            if byte_buf:
                parcalar.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf = bytearray()

        for i in ids:
            tok = self.id_to_kelime.get(int(i), "")
            if not tok or tok in self.OZEL:
                continue
            b = self._byte_degeri(tok)
            if b is not None:
                byte_buf.append(b)
                continue
            flush_bytes()
            parcalar.append(self._normal_fragment(tok))
        flush_bytes()

        metin = "".join(parcalar)
        # Alt-kelime birleştirmesinden kalan boşluklu noktalamayı düzelt.
        for a, b in [(" .", "."), (" ,", ","), (" !", "!"),
                     (" ?", "?"), (" ;", ";"), (" :", ":")]:
            metin = metin.replace(a, b)
        return re.sub(r"\s+", " ", metin).strip()

    def ids_to_text(self, ids) -> str:
        return self.decode(ids)

    # ── Kalıcılık ───────────────────────────────────────────────────────
    def kaydet(self, yol: str):
        data = {
            "baglam_penceresi": self.baglam_penceresi,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "special_tokens": list(self.OZEL),
            "byte_fallback": self.byte_fallback_var,
            "kelime_to_id": self.kelime_to_id,
        }
        with open(yol, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def yukle(self, yol: str) -> "BPETokenizer":
        with open(yol, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.baglam_penceresi = int(data.get("baglam_penceresi", self.baglam_penceresi))
        self.max_vocab_size = int(data.get("max_vocab_size", self.max_vocab_size))
        self.min_freq = int(data.get("min_freq", self.min_freq))
        self.kelime_to_id = {k: int(v) for k, v in data.get("kelime_to_id", {}).items()}
        self.id_to_kelime = {v: k for k, v in self.kelime_to_id.items()}
        self.sozluk_boyutu = len(self.kelime_to_id)
        self.subword_set = set(self.kelime_to_id.keys())
        self._cache.clear()
        self._egitildi = True
        self.vocab_tutarliligi(strict=True)
        return self

    # ── Doğrulama yardımcıları ──────────────────────────────────────────
    def ozel_tokenlari_dogrula(self, embedding_boyutu: Optional[int] = None,
                               strict: bool = False) -> Dict:
        """Special token ID'lerini ve opsiyonel embedding kapsamını kontrol et.

        ``embedding_boyutu`` verilirse embedding matrisinin bütün special
        token ID'lerini kapsayıp kapsamadığı da denetlenir.
        """
        beklenen = {
            "<PAD>": self.PAD_ID,
            "<UNK>": self.UNK_ID,
            "<BOS>": self.BOS_ID,
            "<EOS>": self.EOS_ID,
        }
        hatalar = []
        for tok, idx in beklenen.items():
            if self.kelime_to_id.get(tok) != idx:
                hatalar.append(f"{tok} id={self.kelime_to_id.get(tok)}; beklenen {idx}")
            if self.id_to_kelime.get(idx) != tok:
                hatalar.append(f"id_to_kelime[{idx}]={self.id_to_kelime.get(idx)}; beklenen {tok}")
            if embedding_boyutu is not None and idx >= int(embedding_boyutu):
                hatalar.append(f"embedding boyutu {embedding_boyutu}, {tok}/{idx} için yetersiz")
        rapor = {"ok": not hatalar, "hatalar": hatalar, "beklenen": beklenen}
        if strict and hatalar:
            raise ValueError("Special token doğrulaması başarısız: " + "; ".join(hatalar))
        return rapor

    def vocab_tutarliligi(self, expected_size: Optional[int] = None,
                          embedding_boyutu: Optional[int] = None,
                          strict: bool = False) -> Dict:
        """BPE sözlük boyutu ve ID sürekliliği kontrolü.

        * ``expected_size``: gerçek vocab dosyasının beklenen tam boyutu.
        * ``embedding_boyutu``: model embedding satır sayısı; vocab'tan küçükse
          eğitim sessizce yanlış sonuç verebilir.
        """
        hatalar = []
        if self.sozluk_boyutu != len(self.kelime_to_id):
            hatalar.append(f"sozluk_boyutu={self.sozluk_boyutu}; gerçek={len(self.kelime_to_id)}")
        ids = sorted(self.kelime_to_id.values())
        if ids != list(range(len(ids))):
            hatalar.append("token ID'leri 0..N-1 aralığında süreklilik göstermiyor")
        if len(set(ids)) != len(ids):
            hatalar.append("tekrarlı token ID tespit edildi")
        if expected_size is not None and len(self.kelime_to_id) != int(expected_size):
            hatalar.append(f"vocab boyutu {len(self.kelime_to_id)}; beklenen {expected_size}")
        if embedding_boyutu is not None and int(embedding_boyutu) < len(self.kelime_to_id):
            hatalar.append(f"embedding boyutu {embedding_boyutu}, vocab {len(self.kelime_to_id)} için küçük")
        ozel = self.ozel_tokenlari_dogrula(embedding_boyutu=embedding_boyutu)
        hatalar.extend(ozel["hatalar"])
        rapor = {
            "ok": not hatalar,
            "hatalar": hatalar,
            "sozluk_boyutu": len(self.kelime_to_id),
            "max_vocab_size": self.max_vocab_size,
            "byte_fallback": self.byte_fallback_var,
        }
        if strict and hatalar:
            raise ValueError("Vocab tutarlılığı başarısız: " + "; ".join(hatalar))
        return rapor

    # ══════════════════════════════════════════════════════════════════
    # GeometrikTokenizer API uyumluluğu (BPE artık zincirin varsayılan
    # tokenizer'ıdır — rapor 8.4.6; eski kelime-bazlı tokenizer ile aynı
    # arayüzü konuşur: sozluk / encode / decode / fit / id_to_kelime)
    # ══════════════════════════════════════════════════════════════════
    @property
    def sozluk(self) -> Dict[str, int]:
        """Kelime→id sözlüğü (alt-kelime parçaları)."""
        return self.kelime_to_id

    def encode(self, metin: str, pad: bool = False) -> List[int]:
        """Metni id listesine çevirir. pad=True → sağa hizalı, pencereye sabitlenmiş."""
        return self._ids_from_text(metin, pad=pad)

    # Takma adlar: calistir.py'deki genel "fit" araması için
    fit = fit_on_text
