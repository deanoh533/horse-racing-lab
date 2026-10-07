"""공공데이터포털 한국마사회 API에서 경주성적을 받아 CSV로 저장한다.

    export KRA_API_KEY=발급받은_서비스키(디코딩 키)
    python -m hrl.fetch --endpoint <API 명세서의 요청 주소> --param <조회 조건>=<값> --out ../data/raw/2024-01.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .kra import fetch_kra_api


def main() -> None:
    ap = argparse.ArgumentParser(description="마사회 경주성적 API 수집")
    ap.add_argument("--endpoint", required=True, help="API 요청 주소 (공공데이터포털 명세서 참고)")
    ap.add_argument("--param", action="append", default=[], help="조회 조건 key=value (여러 번 지정 가능)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = os.environ.get("KRA_API_KEY")
    if not key:
        raise SystemExit("환경변수 KRA_API_KEY에 공공데이터포털 서비스키를 넣어 주세요.")
    params = dict(p.split("=", 1) for p in args.param)
    raw = fetch_kra_api(args.endpoint, key, params)
    if raw.empty:
        raise SystemExit("받은 데이터가 없습니다. 조회 조건을 확인하세요.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"{len(raw):,}행 저장: {args.out}")


if __name__ == "__main__":
    main()
