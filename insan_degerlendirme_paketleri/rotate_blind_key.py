"""Kör değerlendirme anahtarını yeni bir salt ile rotasyona tabi tutar.

Kullanım:

    python insan_degerlendirme_paketleri/rotate_blind_key.py \\
        --eski <eski_kor_anahtari.json> \\
        --cikti <yeni_kor_anahtari.json> \\
        [--seed-hex <32-byte-hex>]

Notlar:
- Bu araç yalnız yeni hash'leri üretir. Değerlendirici paketleri bu yeni
  hash'lerle **ayrıca yeniden üretilmelidir**.
- Üretilen dosya repoya commit edilmez; yalnızca güvenli bir kanalda
  saklanır. SHA-256 digest'i `artifacts/blind_key_digest.txt` olarak
  yayımlanabilir.
- Rotasyon sonrası eski hash'ler geçersizdir; eski değerlendirme
  "pre-rotation" olarak etiketlenmelidir.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from pathlib import Path


def yeni_hash(eski_hash: str, seed: bytes) -> str:
    """Eski hash + yeni salt → yeni hash (HMAC-SHA256, ilk 16 hex)."""
    return hmac.new(seed, eski_hash.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="Kör anahtarını rotasyona tabi tut")
    ap.add_argument("--eski", type=Path, required=True,
                    help="Mevcut kor_anahtari.json yolu")
    ap.add_argument("--cikti", type=Path, required=True,
                    help="Yeni anahtar dosyası (repo dışı)")
    ap.add_argument("--seed-hex", type=str, default=None,
                    help="32-byte hex salt (verilmezse rastgele üretilir)")
    ns = ap.parse_args()

    eski = json.loads(ns.eski.read_text(encoding="utf-8"))
    seed = bytes.fromhex(ns.seed_hex) if ns.seed_hex else secrets.token_bytes(32)

    yeni: dict[str, dict] = {}
    for eski_h, meta in eski.items():
        yeni[yeni_hash(eski_h, seed)] = meta

    ns.cikti.write_text(
        json.dumps(yeni, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )

    digest = hashlib.sha256(ns.cikti.read_bytes()).hexdigest()
    print(f"Eski kayıt:  {len(eski)}")
    print(f"Yeni kayıt:  {len(yeni)}")
    print(f"Salt (hex):  {seed.hex()}")
    print(f"Çıktı:       {ns.cikti}")
    print(f"SHA-256:     {digest}")
    print()
    print("Uyarı: Bu dosyayı repoya COMMIT ETMEYİN.")
    print("Yalnızca SHA-256 digest'ini artifacts/ altında yayımlayın.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
