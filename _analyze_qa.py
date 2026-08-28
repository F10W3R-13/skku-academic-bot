"""QA 로그 분석: 답변이 허술했던 질문을 골라 크롤링/보강 후보를 보여준다.

    python _analyze_qa.py [--log-dir logs] [--days 7]

판정 기준(임계값은 api.py 의 검색 설정값과 같은 값을 쓴다):
  gap     검색 최고점수 < MIN_SEED_SCORE  -> 코퍼스에 아예 없는 주제일 가능성이 큼
  weak    MIN_SEED_SCORE <= 점수 < FILL_MIN_SCORE -> 걸리긴 했지만 불확실한 검색
  hedged  답변에 "not sure" 류 불확실 표현 -> 근거가 부족했다는 신호
  empty   모델이 빈 답변을 반환 (출력 예산 부족 등)
  error   /ask 자체가 실패

같은 질문(대소문자·공백 정규화)은 하나로 묶어 횟수와 최고 점수를 보여준다.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# 진단 도구는 의존성이 깨져 있어도 돌아야 하므로, 못 가져오면 기본값으로.
try:
    from api import FILL_MIN_SCORE, MIN_SEED_SCORE
except Exception:  # noqa: BLE001
    MIN_SEED_SCORE = 0.45
    FILL_MIN_SCORE = 0.54

HEDGE_MARKERS = ("not sure", "don't know", "do not know", "모르겠", "확실하지 않")


def load_records(log_dir: Path, days: int) -> list[dict]:
    records = []
    cutoff = datetime.now().astimezone() - timedelta(days=days) if days else None
    for path in sorted(log_dir.glob("qa_*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] 깨진 로그 줄 건너뜀: {path.name}:{line_no}", file=sys.stderr)
                continue
            if cutoff:
                try:
                    if datetime.fromisoformat(rec["ts"]) < cutoff:
                        continue
                except (KeyError, ValueError):
                    pass  # ts 를 못 읽으면 기간 필터 없이 포함
            records.append(rec)
    return records


def classify(rec: dict) -> list[str]:
    if rec.get("status") != "ok":
        return ["error"]
    flags = []
    top = (rec.get("retrieval") or {}).get("top_score")
    if top is None or top < MIN_SEED_SCORE:
        flags.append("gap")
    elif top < FILL_MIN_SCORE:
        flags.append("weak")
    answer = (rec.get("answer") or "").lower()
    if any(m in answer for m in HEDGE_MARKERS):
        flags.append("hedged")
    if not (rec.get("answer") or "").strip():
        flags.append("empty")
    if rec.get("retrieval", {}).get("fallback"):
        flags.append("fallback")
    return flags or ["ok"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", default="logs", help="QA 로그 폴더 (기본: logs)")
    ap.add_argument("--days", type=int, default=0, help="최근 N일만 (0=전체)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"로그 폴더가 없습니다: {log_dir}")
        return 1
    records = load_records(log_dir, args.days)
    if not records:
        print("분석할 로그가 없습니다.")
        return 0

    flag_counts: Counter[str] = Counter()
    # 정규화된 질문 -> {flags, count, best_top_score, last_ts, example}
    questions: dict[str, dict] = {}
    for rec in records:
        flags = classify(rec)
        flag_counts.update(flags)
        if flags == ["ok"]:
            continue
        q = " ".join(rec.get("question", "").split()).casefold()
        top = (rec.get("retrieval") or {}).get("top_score")
        entry = questions.setdefault(
            q, {"flags": set(), "count": 0, "best": None, "last_ts": "", "question": rec.get("question", "")}
        )
        entry["flags"].update(flags)
        entry["count"] += 1
        if top is not None and (entry["best"] is None or top > entry["best"]):
            entry["best"] = top
        if rec.get("ts", "") > entry["last_ts"]:
            entry["last_ts"] = rec.get("ts", "")

    total = len(records)
    print(f"QA 로그 {total}건 ({log_dir}), 임계값: gap<{MIN_SEED_SCORE} <= weak <{FILL_MIN_SCORE}")
    for name in ("ok", "gap", "weak", "hedged", "empty", "fallback", "error"):
        if flag_counts.get(name):
            print(f"  {name:<8} {flag_counts[name]}")

    order = {"gap": 0, "weak": 1, "hedged": 2, "empty": 3, "fallback": 4, "error": 5}
    def severity(entry):
        return min(order[f] for f in entry["flags"])

    flagged = sorted(questions.values(), key=severity)
    groups = [("gap", "코퍼스에 없는 주제로 보이는 질문 (크롤링 1순위)"),
              ("weak", "불확실한 검색으로 답변된 질문"),
              ("hedged", "답변이 불확실 표현을 포함한 질문"),
              ("empty", "빈 답변이 나온 질문"),
              ("fallback", "하한 미달로 폴백 검색된 질문"),
              ("error", "답변 실패한 질문")]
    for flag, title in groups:
        rows = [e for e in flagged if flag in e["flags"]]
        if not rows:
            continue
        rows.sort(key=lambda e: -e["count"])
        print(f"\n[{title}]")
        for e in rows:
            best = f"{e['best']:.3f}" if e["best"] is not None else "  -  "
            print(f"  x{e['count']:<3} top={best}  {e['question'][:80]}  ({e['last_ts']})")

    if not flagged:
        print("\n보강이 필요한 질문이 없습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
