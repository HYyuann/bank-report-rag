#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四周作业（方向 A）第一步：从巨潮资讯网批量下载 A 股银行 2026 年半年度报告 PDF。

巨潮公告查询接口：POST http://www.cninfo.com.cn/new/hisAnnouncement/query
  - stock    必须写成 "代码,orgId"（只给代码会返回 0 条），orgId 由 topSearch 接口换取
  - column   深市 szse / 沪市 sse，必须与上市地一致
  - category category_bndbg_szsh 即「半年度报告」，沪深通用
  - seDate   公告日期区间，形如 2026-01-01~2026-12-31

用法：
    .venv/Scripts/python.exe download_reports.py              # 试点：招商银行 + 平安银行
    .venv/Scripts/python.exe download_reports.py --all        # 全部 12 家
    .venv/Scripts/python.exe download_reports.py --codes 600036,601398
    .venv/Scripts/python.exe download_reports.py --all --force  # 忽略已存在文件重新下载
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from pypdf import PdfReader

# Windows 控制台/管道默认走 cp936，中文会乱码；统一固定成 UTF-8，
# 免去每次手动 chcp 65001。交互式控制台本就按 Unicode 直写，不受影响。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
MANIFEST_PATH = ROOT / "manifest.json"

REPORT_PERIOD = "2026H1"
SE_DATE = "2026-01-01~2026-12-31"
CATEGORY = "category_bndbg_szsh"  # 半年度报告

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
ORG_URL = "http://www.cninfo.com.cn/new/information/topSearch/detailOfQuery"
STATIC_BASE = "http://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
}

# (代码, 简称, 上市地 column)
BANKS = [
    ("601398", "工商银行", "sse"),
    ("601939", "建设银行", "sse"),
    ("601288", "农业银行", "sse"),
    ("601988", "中国银行", "sse"),
    ("601328", "交通银行", "sse"),
    ("600036", "招商银行", "sse"),
    ("601166", "兴业银行", "sse"),
    ("600000", "浦发银行", "sse"),
    ("601998", "中信银行", "sse"),
    ("600016", "民生银行", "sse"),
    ("601818", "光大银行", "sse"),
    ("000001", "平安银行", "szse"),
]

PILOT_CODES = ["600036", "000001"]

# 探测阶段实测到的 orgId，接口临时抽风时兜底
ORG_ID_FALLBACK = {
    "601398": "jjxt0000019",
    "601939": "9900003682",
    "601288": "jjxt0000020",
    "601988": "jjxt0000028",
    "601328": "9900002841",
    "600036": "gssh0600036",
    "601166": "9900002081",
    "600000": "gssh0600000",
    "601998": "9900002721",
    "600016": "gssh0600016",
    "601818": "9900006246",
    "000001": "gssz0000001",
}

# 标题必须命中其一，否则不是半年报正文
REQUIRE_KEYWORDS = ("半年度报告", "半年报")
# 命中任一即排除：摘要 / 英文版 / 更正补充 / 提示性公告 / 取消修订 / 问询回复
EXCLUDE_KEYWORDS = (
    "摘要", "英文", "English", "更正", "补充", "取消", "废止", "修订",
    "提示性", "公告", "说明", "问询", "反馈", "H股",
)

BJT = timezone(timedelta(hours=8))


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    print(msg, flush=True)


def pace(lo: float = 1.0, hi: float = 2.0) -> None:
    """请求间隔限速，默认 1~2 秒。"""
    time.sleep(random.uniform(lo, hi))


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def ts_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=BJT).strftime("%Y-%m-%d")


def human_size(n: int) -> str:
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


# --------------------------------------------------------------------------- #
# 巨潮接口
# --------------------------------------------------------------------------- #
def resolve_org_id(session: requests.Session, code: str) -> str:
    """用股票代码换 orgId。"""
    try:
        resp = session.post(
            ORG_URL,
            data={"keyWord": code, "maxSecNum": 10, "maxListNum": 10},
            timeout=30,
        )
        resp.raise_for_status()
        for item in resp.json().get("keyBoardList") or []:
            if item.get("code") == code and item.get("category") == "A股":
                return item["orgId"]
    except Exception as exc:  # noqa: BLE001 - 兜底优先
        log(f"   ! orgId 查询失败（{exc}），改用内置映射")

    if code in ORG_ID_FALLBACK:
        return ORG_ID_FALLBACK[code]
    raise RuntimeError(f"{code}: 既未取到 orgId，也无内置兜底值")


def query_announcements(
    session: requests.Session, code: str, org_id: str, column: str
) -> list[dict]:
    """查询该股票 2026 年的半年度报告类公告，翻页取全。"""
    records: list[dict] = []
    page = 1
    while True:
        resp = session.post(
            QUERY_URL,
            data={
                "pageNum": page,
                "pageSize": 30,
                "column": column,
                "tabName": "fulltext",
                "plate": "",
                "stock": f"{code},{org_id}",
                "searchkey": "",
                "secid": "",
                "category": CATEGORY,
                "trade": "",
                "seDate": SE_DATE,
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("announcements") or []
        records.extend(batch)
        if not payload.get("hasMore") or not batch:
            break
        page += 1
        pace()

    return [
        {
            "title": strip_tags(a.get("announcementTitle", "")),
            "date": ts_to_date(a["announcementTime"]),
            "url": STATIC_BASE + a["adjunctUrl"].lstrip("/"),
            "size_kb": a.get("adjunctSize") or 0,
        }
        for a in records
    ]


def pick_full_report(records: list[dict]) -> tuple[dict | None, list[dict]]:
    """从公告里挑出半年度报告正文，返回 (正文, 被排除的候选)。"""
    kept, dropped = [], []
    for rec in records:
        title = rec["title"]
        hit_require = any(k in title for k in REQUIRE_KEYWORDS)
        hit_exclude = next((k for k in EXCLUDE_KEYWORDS if k in title), None)
        if hit_require and not hit_exclude and "2026" in title:
            kept.append(rec)
        else:
            rec = {**rec, "skip_reason": hit_exclude or "标题不含半年报关键词"}
            dropped.append(rec)

    # 同日优先取最新的，再按体积取最大（正文远大于摘要）
    kept.sort(key=lambda r: (r["date"], r["size_kb"]), reverse=True)
    return (kept[0] if kept else None), dropped


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download_pdf(
    session: requests.Session, url: str, dest: Path, retries: int = 3
) -> int:
    """流式下载到 .part 临时文件，成功后原子替换。返回字节数。"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=90) as resp:
                resp.raise_for_status()
                written = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            if written < 10240:
                raise ValueError(f"文件仅 {written} 字节，疑似错误页")
            tmp.replace(dest)
            return written
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            wait = 2 ** attempt
            log(f"   ! 第 {attempt} 次下载失败：{exc}；{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def count_pages(path: Path) -> int | None:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        return len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        log(f"   ! 页数解析失败：{exc}")
        return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_manifest() -> dict[str, dict]:
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            return {row["code"]: row for row in data.get("reports", [])}
        except Exception:  # noqa: BLE001
            pass
    return {}


def save_manifest(entries: dict[str, dict]) -> None:
    ordered = [entries[c] for c, _, _ in BANKS if c in entries]
    ordered += [v for k, v in entries.items() if k not in {c for c, _, _ in BANKS}]
    payload = {
        "generated_at": datetime.now(BJT).isoformat(timespec="seconds"),
        "report_period": "2026-06-30",
        "source": "巨潮资讯网 (cninfo.com.cn)",
        "count": len(ordered),
        "reports": ordered,
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def process_bank(
    session: requests.Session,
    code: str,
    name: str,
    column: str,
    force: bool,
) -> dict | None:
    log(f"\n[{code} {name}]")
    org_id = resolve_org_id(session, code)
    log(f"   orgId = {org_id}")

    records = query_announcements(session, code, org_id, column)
    log(f"   命中 {len(records)} 条半年度报告类公告")
    report, dropped = pick_full_report(records)
    for item in dropped:
        log(f"   - 跳过：{item['title']}（{item['date']}，{item['skip_reason']}）")

    if report is None:
        log("   ! 未找到半年度报告正文")
        return None

    log(f"   选中：{report['title']}（{report['date']}，{report['size_kb']} KB）")

    filename = f"{code}_{name}_{REPORT_PERIOD}.pdf"
    dest = FILES_DIR / filename

    if dest.exists() and not force:
        log(f"   已存在，跳过下载：{filename}")
    else:
        size = download_pdf(session, report["url"], dest)
        log(f"   下载完成：{filename}（{human_size(size)}）")
        pace()

    pages = count_pages(dest)
    return {
        "code": code,
        "name": name,
        "title": report["title"],
        "announce_date": report["date"],
        "url": report["url"],
        "filename": filename,
        "size_bytes": dest.stat().st_size,
        "pages": pages,
        "downloaded_at": datetime.now(BJT).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="巨潮银行半年报批量下载")
    parser.add_argument("--codes", help="逗号分隔的股票代码，如 600036,000001")
    parser.add_argument("--all", action="store_true", help="下载全部 12 家")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    if args.all:
        targets = list(BANKS)
    elif args.codes:
        wanted = [c.strip() for c in args.codes.split(",") if c.strip()]
        lookup = {c: (c, n, col) for c, n, col in BANKS}
        missing = [c for c in wanted if c not in lookup]
        if missing:
            log(f"! 未知代码：{', '.join(missing)}")
            return 2
        targets = [lookup[c] for c in wanted]
    else:
        pilot = set(PILOT_CODES)
        targets = [b for b in BANKS if b[0] in pilot]

    FILES_DIR.mkdir(exist_ok=True)
    entries = load_manifest()

    log("=" * 72)
    log(f"巨潮资讯网 · 2026 年半年度报告下载（{len(targets)} 家）")
    log("=" * 72)

    session = requests.Session()
    session.headers.update(HEADERS)

    ok, failed = [], []
    for idx, (code, name, column) in enumerate(targets):
        try:
            entry = process_bank(session, code, name, column, args.force)
            if entry:
                entries[code] = entry
                ok.append(entry)
            else:
                failed.append((code, name, "未找到正文"))
        except Exception as exc:  # noqa: BLE001
            log(f"   ! 失败：{exc}")
            failed.append((code, name, str(exc)))
        if idx < len(targets) - 1:
            pace()

    save_manifest(entries)

    log("\n" + "=" * 72)
    log("下载结果")
    log("=" * 72)
    log(f"{'代码':<8}{'简称':<10}{'文件名':<28}{'大小':>10}{'页数':>7}  公告日期")
    log("-" * 72)
    for row in ok:
        pages = row["pages"] if row["pages"] is not None else "-"
        log(
            f"{row['code']:<8}{row['name']:<10}{row['filename']:<28}"
            f"{human_size(row['size_bytes']):>10}{str(pages):>7}  {row['announce_date']}"
        )
    if failed:
        log("\n未完成：")
        for code, name, why in failed:
            log(f"  {code} {name}：{why}")

    log(f"\n成功 {len(ok)} / {len(targets)}，文件目录：{FILES_DIR}")
    log(f"manifest：{MANIFEST_PATH}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
