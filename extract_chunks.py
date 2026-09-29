#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四周作业（方向 A）第二步：PDF 文本 + 表格提取 -> 繁简统一 -> 章节识别 -> 切块。

提取策略
--------
正文  PyMuPDF (fitz)，对 xref 损坏的 PDF 容错更好（建行/农行/交行那三份）。
      横排页（/Rotate 90）要单独处理：fitz 的 get_text 不理会页面旋转，给的是
      「内容坐标」，那一页的文字在内容坐标里是竖排的（dir=(0,-1)），表格会碎成
      「一行一个数字」。统一乘 page.rotation_matrix 映射回显示坐标。
表格  pdfplumber，级联两级：
        1) page.find_tables()（lines 策略）—— 有完整框线的表它做得很好；
        2) 若检出表的 bbox 覆盖不到正文宽度（说明丢了列，实测招行/建行
           这类「无框线 + 多列」表会整列丢行标签），则改用 pdfplumber 的
           extract_words() + 坐标聚类重建：数字右对齐的 x1 聚类成列锚点，
           其余归标签列，再按 y 聚类成行。
     列数 < 2 的「表」一律丢弃：pdfplumber 在纯文字页上会把整段正文框成
     1 行×1 列（实测浦发 330 页报出 467 张这种假表）。
     逐张记录走了哪条路，写进 extraction_report.md。

切块  以「原子段落 / 表格行」为单位做滑动窗口，窗口只落在原子边界上，
      因此不会把句子或表格行切成两半；表格若被拆开，续块自动补表头。

落盘
----
逐家落盘 + 可续跑：每份处理完立刻原子写入 extract/{代码}.json，再处理下一份；
启动时扫描 extract/，已有的代码直接跳过；最后把 extract/*.json 按代码排序
合并成 chunks.jsonl。中途被打断也不会留下写了一半的文件。

用法
----
    .venv/Scripts/python.exe extract_chunks.py                # 全部 12 家（自动续跑）
    .venv/Scripts/python.exe extract_chunks.py --codes 600036 # 只跑某几家
    .venv/Scripts/python.exe extract_chunks.py --force        # 忽略已有结果重跑
    .venv/Scripts/python.exe extract_chunks.py --merge-only   # 只合并+出报告
    .venv/Scripts/python.exe extract_chunks.py --sample 5     # 抽样自检
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitz
import pdfplumber
import tiktoken
import zhconv

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
MANIFEST_PATH = ROOT / "manifest.json"
EXTRACT_DIR = ROOT / "extract"          # 逐家中间产物，处理完一家立刻落盘
CHUNKS_PATH = ROOT / "chunks.jsonl"
REPORT_PATH = ROOT / "extraction_report.md"

REPORT_PERIOD = "2026H1"
TARGET_TOKENS = 1200
OVERLAP_TOKENS = 150
SUSPICIOUS_CHARS = 100        # 单页字数低于此值 -> 可疑
NARROW_COVERAGE = 0.55        # 表格 bbox 宽度 / 正文宽度 低于此值 -> 判定丢列，重建
COL_SNAP_TOL = 3.0            # 数字右对齐到列锚点的容差（磅）
ROW_TOL = 4.0                 # 行聚类容差（磅）

ENC = tiktoken.get_encoding("cl100k_base")
BJT = timezone(timedelta(hours=8))

# 章节归一化：关键词 -> 规范章节名（按优先级从具体到宽泛）
CANONICAL = [
    ("管理层讨论与分析", ("管理层讨论与分析", "经营情况讨论与分析", "讨论与分析", "管理层讨论及分析")),
    ("财务报表附注", ("财务报表附注", "财务报告附注", "会计报表附注")),
    ("财务报告", ("财务报告", "财务报表", "审阅报告", "中期财务报表")),
    ("公司治理", ("公司治理",)),
    ("重要事项", ("重要事项", "重大事项")),
    ("股份变动及股东情况", ("股份变动", "股东情况", "董事、高级管理人员", "董事、监事")),
    ("债券相关情况", ("债券相关情况", "债券情况")),
    ("环境、社会与治理", ("环境、社会与治理", "环境和社会", "社会责任", "ESG")),
    ("公司简介与主要财务指标", (
        "公司简介", "本行简介", "主要财务指标", "公司基本情况", "会计数据和财务指标",
        "财务数据和财务指标", "财务摘要", "主要会计数据",
    )),
    ("重要提示与释义", ("重要提示", "释义", "备查文件")),
]
UNKNOWN_SECTION = "未识别"

# 「第X节 / 第X章」标题行
HEAD_CHAPTER = re.compile(r"^第\s*[一二三四五六七八九十百]+\s*[节章]")
# 只有「第X节」而标题在下一行（兴业/光大等家目录与正文都这样排版）
CHAPTER_ONLY = re.compile(r"^第\s*[一二三四五六七八九十百]+\s*[节章]\s*$")
# 目录行（用于剔除目录页）
TOC_LINE = re.compile(r"[.·]{4,}\s*\d+\s*$")
# 点线引导符（目录行特征），剥掉后才能拿到干净的标题
DOT_LEADER = re.compile(r"[.·…]{3,}")
# 文档标题类书签（如「…2026年半年度报告_定稿」），不是章节，须排除
DOC_TITLE = re.compile(r"半年度报告|年度报告|季度报告|上传稿|定稿|正文\)|_定稿")
NUMERIC = re.compile(r"^[\d,.\-()%<>≤≥]+$")
# 标题前的编号（「1 财务摘要」「3.2 业务回顾」）；不剥掉的话关键词匹配认不出来
TITLE_NUM = re.compile(r"^\d+(?:[.\-、]\d+)*[.、]?")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    print(msg, flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    """写临时文件再原子改名，避免写到一半被打断留下残缺文件。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def extract_path(code: str) -> Path:
    return EXTRACT_DIR / f"{code}.json"


def clean_text(text: str) -> str:
    text = zhconv.convert(text, "zh-cn")
    text = text.replace("　", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def n_tokens(text: str) -> int:
    return len(ENC.encode(text)) if text else 0


def canonical_section(raw: str) -> str:
    flat = re.sub(r"\s+", "", raw).lstrip("|")   # 合并行可能带前导「| 」
    for canon, keys in CANONICAL:
        if any(k in flat for k in keys):
            return canon
    return raw.strip().lstrip("|").strip() or UNKNOWN_SECTION


# --------------------------------------------------------------------------- #
# 表格：级联提取
# --------------------------------------------------------------------------- #
def _cluster(values: list[float], tol: float) -> list[float]:
    """一维聚类，返回簇中心（升序）。"""
    if not values:
        return []
    vals = sorted(values)
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [sum(g) / len(g) for g in groups]


def rebuild_table_by_words(words: list[tuple]) -> list[list[str]] | None:
    """
    用词坐标重建表格。
    words: [(x0, y0, x1, y1, text), ...]
    数字右对齐 -> 按 x1 聚类成列锚点；其余归到标签列；再按 y 聚类成行。
    """
    if not words:
        return None

    anchors = _cluster([w[2] for w in words if NUMERIC.match(w[4])], COL_SNAP_TOL)
    if len(anchors) < 1:
        return None
    n_cols = len(anchors) + 1  # 第 0 列是标签列

    # 按行聚类
    rows_map: dict[float, list[tuple]] = defaultdict(list)
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        yc = (w[1] + w[3]) / 2
        # 找到已存在的、中心距 <= ROW_TOL 的行
        target = None
        for key in rows_map:
            if abs(key - yc) <= ROW_TOL:
                target = key
                break
        rows_map[target if target is not None else yc].append(w)

    cells_rows: list[list[str]] = []
    for y in sorted(rows_map):
        row = rows_map[y]
        cells: list[list[str]] = [[] for _ in range(n_cols)]
        for w in sorted(row, key=lambda w: w[0]):
            ci = 0
            for k, a in enumerate(anchors):
                if abs(w[2] - a) <= COL_SNAP_TOL:
                    ci = k + 1
                    break
            cells[ci].append(w[4])
        cells_rows.append([" ".join(c).strip() for c in cells])

    # 合并「只有标签列」的被折行：并入下一个有数据的行
    merged: list[list[str]] = []
    pending: list[str] = []
    for r in cells_rows:
        has_data = any(c for c in r[1:])
        if not has_data and r[0]:
            pending.append(r[0])
            continue
        if pending:
            r = list(r)
            r[0] = "".join(pending) + r[0]
            pending = []
        merged.append(r)
    if pending:
        if merged:
            merged[-1][0] = merged[-1][0] + "".join(pending)
        else:
            merged.append(["".join(pending)] + [""] * (n_cols - 1))

    merged = [r for r in merged if any(c for c in r)]
    if len(merged) < 2 or n_cols < 2:
        return None
    return merged


def rot_bbox(bbox, rot_m):
    """把内容坐标的 bbox 映射到显示坐标（横排页要乘 rotation_matrix）。"""
    if rot_m is None:
        return tuple(bbox[:4])
    x0, y0, x1, y1 = bbox[:4]
    xs, ys = [], []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        xs.append(rot_m.a * x + rot_m.c * y + rot_m.e)
        ys.append(rot_m.b * x + rot_m.d * y + rot_m.f)
    return (min(xs), min(ys), max(xs), max(ys))


def detect_tables(fitz_page, pl_page, rot_m=None) -> list[dict]:
    """返回 [{bbox, rows, method}]，method ∈ pdfplumber / rebuilt / pdfplumber-narrow。"""
    try:
        words = [(*rot_bbox(w, rot_m), *w[4:]) for w in fitz_page.get_text("words")]
    except Exception:  # noqa: BLE001
        words = []
    if words:
        text_left = min(w[0] for w in words)
        text_right = max(w[2] for w in words)
    else:
        text_left, text_right = 0.0, fitz_page.rect.width
    text_width = max(text_right - text_left, 1.0)

    try:
        found = pl_page.find_tables()
    except Exception:  # noqa: BLE001
        return []

    out = []
    for tb in found:
        bbox = rot_bbox(tb.bbox, rot_m)
        try:
            rows = tb.extract()
        except Exception:  # noqa: BLE001
            rows = []
        rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r] for r in rows]

        # 单列「表」不是表：pdfplumber 在纯文字页上会把整段正文框成 1 行×1 列
        # （实测浦发 330 页报出 467 张），caption 兜底又取到页眉，结果每段正文前
        # 都粘一句「[表头] 2026 年半年度报告」。列数 < 2 一律丢弃，交给正文通道。
        if max((len(r) for r in rows), default=0) < 2:
            continue

        width = bbox[2] - bbox[0]
        coverage = width / text_width

        if coverage >= NARROW_COVERAGE:
            out.append({"bbox": bbox, "rows": rows, "method": "pdfplumber", "coverage": coverage})
            continue

        # 覆盖不足 -> 判定丢列，用同 y 区间的词重建
        band = [w for w in words if w[1] >= bbox[1] - 2 and w[3] <= bbox[3] + 2]
        rebuilt = rebuild_table_by_words(band)
        # 只按「恢复了多少非空单元格」判优劣：原表的空列计数没有意义，
        # 拿列数去比会把重建结果误判为更差（实测建行 p16 就是这样被否掉的）。
        if rebuilt and n_filled(rebuilt) > n_filled(rows):
            out.append({"bbox": bbox, "rows": rebuilt, "method": "rebuilt", "coverage": coverage})
        else:
            out.append({"bbox": bbox, "rows": rows, "method": "pdfplumber-narrow", "coverage": coverage})
    return out


def render_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(c for c in r) for r in rows)


def n_filled(rows: list[list[str]]) -> int:
    """非空单元格数，用于判断重建结果是否真的比原表信息更多。"""
    return sum(1 for r in rows for c in r if str(c).strip())


def split_header(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """
    拆出表头：第一行「含 >=2 个纯数字单元格」的行的前面都算表头。
    银行财报常见 2~4 层合并表头（如建行那种「截至…止六个月 / 平均余额 / 利息支出」），
    这样能把多层表头整体保留下来，而不是只留第一行。
    """
    for k, row in enumerate(rows):
        if sum(1 for c in row if NUMERIC.match(str(c).strip())) >= 2:
            return rows[:k], rows[k:]
    return [], rows


# --------------------------------------------------------------------------- #
# 章节识别
# --------------------------------------------------------------------------- #
def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 30:
        return False
    if HEAD_CHAPTER.match(s):
        return True
    # 目录行剥掉点线引导符和尾部页码后，才可能是干净标题
    if DOT_LEADER.search(s):
        s = re.sub(r"[.·…]+\s*\d*\s*$", "", s)
    flat = TITLE_NUM.sub("", re.sub(r"\s+", "", s)).lstrip("|")
    if not flat or len(flat) > 30:
        return False
    for canon, keys in CANONICAL:
        for k in keys:
            if flat.startswith(k):
                return True
    return False


def bookmark_page_sections(doc, n_pages: int) -> list[str]:
    """PDF 书签 -> 每页章节名（1、2 级标题），无书签则返回全 UNKNOWN。"""
    page_sec = [UNKNOWN_SECTION] * n_pages
    toc = doc.get_toc()
    if not toc:
        return page_sec
    cur = UNKNOWN_SECTION
    marks = sorted(
        (
            (max(pg - 1, 0), canonical_section(title))
            for lvl, title, pg in toc
            if lvl <= 2 and not DOC_TITLE.search(title)   # 跳过文档标题类书签
        ),
        key=lambda x: x[0],
    )
    for i in range(n_pages):
        while marks and marks[0][0] <= i:
            cur = marks.pop(0)[1]
        page_sec[i] = cur
    return page_sec


def detect_sections(doc, pages_lines: list[list[str]]) -> tuple[list[list[str]], list[str], str]:
    """
    返回 (逐行章节, 页级书签章节, 采用的方式)。
    行级识别粒度更细，作主；书签作逐块兜底填补「未识别」的块。
    """
    # 目录页：标题行密集或点线引导符密集
    toc_pages = set()
    for i, lines in enumerate(pages_lines):
        n_head = sum(1 for l in lines if is_heading(l))
        # 用「点线引导符」而不是「点线+页码」判目录页：工行的目录点线被排版切成
        # 「... ...」（点之间有空格），TOC_LINE 要求结尾是页码，匹配不上；目录页
        # 漏判后页面里的「释义/公司治理」等条目会被当成正文标题，把后面几十页
        # 全都带偏（实测工行 12 个块被误标成「重要提示与释义」）。
        n_dots = sum(1 for l in lines if DOT_LEADER.search(l))
        if n_head >= 4 or n_dots >= 4:
            toc_pages.add(i)

    per_line_section: list[list[str]] = []
    cur = UNKNOWN_SECTION          # 章节状态跨页保持，不能每页重置
    for i, lines in enumerate(pages_lines):
        page_sections = []
        for j, line in enumerate(lines):
            if i not in toc_pages and is_heading(line):
                title = line.strip()
                # 「第一节」独占一行时，标题在下一行
                if CHAPTER_ONLY.match(title) and j + 1 < len(lines):
                    title = f"{title} {lines[j + 1].strip()}"
                cur = canonical_section(
                    re.sub(r"^第\s*[一二三四五六七八九十百]+\s*[节章]", "", title)
                )
            page_sections.append(cur)
        per_line_section.append(page_sections)

    bookmark_sec = bookmark_page_sections(doc, len(pages_lines))

    n_line = len({s for p in per_line_section for s in p} - {UNKNOWN_SECTION})
    n_book = len(set(bookmark_sec) - {UNKNOWN_SECTION})
    if n_line >= 3 and n_book >= 3:
        method = "标题正则+书签兜底"
    elif n_line >= 3:
        method = "标题正则"
    elif n_book >= 3:
        method = "PDF 书签"
    else:
        method = "未识别"
    return per_line_section, bookmark_sec, method


# --------------------------------------------------------------------------- #
# 单份报告
# --------------------------------------------------------------------------- #
def process_report(code: str, name: str, filename: str, manifest_pages: int | None) -> dict:
    path = FILES_DIR / filename
    log(f"\n[{code} {name}] {filename}")

    doc = fitz.open(path)
    n_pages = len(doc)
    page_stats: list[dict] = []
    table_methods: Counter = Counter()
    rot_pages = 0
    n_tables = 0
    pages_elems: list[list[dict]] = []

    with pdfplumber.open(path) as plumber:
        pages_lines: list[list[str]] = []
        pages_meta: list[dict] = []

        for i in range(n_pages):
            fpage = doc[i]
            try:
                pl_page = plumber.pages[i]
            except Exception:  # noqa: BLE001
                pl_page = None

            # fitz 的 get_text 不理会页面 /Rotate，给的是「内容坐标」；横排页
            # （/Rotate 90，实测交行 38 页、招行 4 页、建行/中行各 1 页）在内容
            # 坐标里文字是竖着排的（dir=(0,-1)），表格行会碎成一行一个数字。
            # 乘 rotation_matrix 映射回显示坐标，才是读者看到的那一版。
            rot_m = fpage.rotation_matrix if fpage.rotation else None
            if rot_m is not None:
                rot_pages += 1

            tables = detect_tables(fpage, pl_page, rot_m) if pl_page is not None else []
            tboxes = [t["bbox"] for t in tables]

            # 正文：取所有「行」，剔除落在表格 bbox 内的；再把同一视觉行
            # （同一 y 中心）上的多段用 " | " 合并。无框线表格 pdfplumber 检不出，
            # 其单元格会各自成行，不合并就会碎成「一行一个数字」，检索时不可用。
            elems = []
            try:
                blocks = [b for b in fpage.get_text("dict")["blocks"] if b.get("type") == 0]
            except Exception as exc:  # noqa: BLE001
                blocks = []
                page_stats.append({"page": i + 1, "chars": 0, "reason": f"正文提取异常: {exc}"})

            # 双栏排版检测：同一 y 段上有左右两块、x 方向明显分开、且每块自身够宽
            # （真正文栏 ≈ 半页宽，实测兴业双栏页 216pt）。窄条要被挡掉 —— 无框线
            # 表格的单元格各自成块时也是「同 y、x 分开」（实测交行每列仅 12-30pt），
            # 不排除就会把表格页误标成双栏。注意不能要求「块内多行」：双栏页每块
            # 恰恰只有一行。
            cand = []
            for b in blocks:
                bx = rot_bbox(b["bbox"], rot_m)
                if (bx[2] - bx[0]) <= 100:
                    continue
                if any(ox0 - 1 <= (bx[0] + bx[2]) / 2 <= ox2 + 1
                       and oy0 - 1 <= (bx[1] + bx[3]) / 2 <= oy2 + 1
                       for ox0, oy0, ox2, oy2 in tboxes):
                    continue
                cand.append(bx)
            multicol = False
            if len(cand) >= 4:
                mid = fpage.rect.width / 2
                left = [c for c in cand if (c[0] + c[2]) / 2 < mid]
                right = [c for c in cand if (c[0] + c[2]) / 2 >= mid]
                if left and right:
                    gutter = min(c[0] for c in right) - max(c[2] for c in left)
                    cover_l = max(c[3] for c in left) - min(c[1] for c in left)
                    cover_r = max(c[3] for c in right) - min(c[1] for c in right)
                    # 中间有明确栏沟，且两栏各自纵向铺满大半页
                    multicol = gutter > 10 and cover_l > 200 and cover_r > 200

            def block_lines(b) -> list[tuple]:
                """块内所有未被表格 bbox 覆盖的行 -> (y中心, x起点, 文本)。"""
                got = []
                for line in b.get("lines", []):
                    lx0, ly0, lx1, ly1 = rot_bbox(line["bbox"], rot_m)
                    cy = (ly0 + ly1) / 2
                    if any(bx0 - 1 <= (lx0 + lx1) / 2 <= bx2 + 1 and by0 - 1 <= cy <= by2 + 1
                           for bx0, by0, bx2, by2 in tboxes):
                        continue
                    text = clean_text("".join(s["text"] for s in line["spans"]))
                    if text:
                        got.append((cy, lx0, text))
                return got

            def emit(group: list[tuple]) -> None:
                """同一视觉行（y 中心接近）上的多段用 " | " 合并成一行。

                无框线表格 pdfplumber 检不出，其单元格会各自成行，不合并就碎成
                「一行一个数字」，检索时不可用。
                """
                rows: list[list] = []
                for cy, lx0, text in group:
                    if rows and abs(rows[-1][0] - cy) <= ROW_TOL:
                        rows[-1][2].append((lx0, text))
                    else:
                        rows.append([cy, lx0, [(lx0, text)]])
                for cy, lx0, segs in rows:
                    elems.append({
                        "kind": "text", "y": cy, "x": lx0, "multicol": multicol,
                        "text": " | ".join(t for _, t in sorted(segs, key=lambda s: s[0])),
                    })

            if multicol:
                # 双栏页：只做块内合并，并保持块的先后（左栏读完再读右栏）。
                # 左右栏分属不同 block 却 y 相同，跨块按 y 合并会把两栏交错成乱码。
                for b in blocks:
                    emit(block_lines(b))
            else:
                # 单栏页：所有块的行按 (y, x) 全局排序后再合并。表格的单元格常
                # 分属不同块（实测交行股东权益变动表，每一列各成一块），只在块内
                # 合并不管用；排序同时把段落顺序也修正回阅读顺序。
                every = [t for b in blocks for t in block_lines(b)]
                every.sort(key=lambda t: (t[0], t[1]))
                emit(every)

            # 表格按其 y 位置插进正文之间，正文本身维持 block 顺序
            for t in tables:
                n_tables += 1
                table_methods[t["method"]] += 1
                elem = {
                    "kind": "table",
                    "y": t["bbox"][1],
                    "x": t["bbox"][0],
                    "text": render_table(t["rows"]),
                    "rows": t["rows"],
                    "table_id": n_tables,
                }
                pos = len(elems)
                for k, e in enumerate(elems):
                    if e["y"] > t["bbox"][1]:
                        pos = k
                        break
                elems.insert(pos, elem)

            # 表格紧邻的上一段正文末行，常是表标题/表头（如「2.2 负债结构」、
            # 「2026年6月30日 2025年12月31日」），用作该表的表头兜底。
            prev_text = ""
            for e in elems:
                if e["kind"] == "text":
                    ls = e["text"].splitlines()
                    prev_text = ls[-1].strip() if ls else ""
                else:
                    # 页眉（「2026 年半年度报告」之类）不是表标题，别传给表当表头
                    e["caption"] = "" if DOC_TITLE.search(prev_text) else prev_text

            lines: list[str] = []
            for e in elems:
                e["line_start"] = len(lines)
                if e["kind"] == "text":
                    lines.extend(e["text"].splitlines())
                else:
                    lines.extend(" | ".join(r) for r in e["rows"])
            pages_lines.append(lines)
            pages_elems.append(elems)
            pages_meta.append({"page": i + 1, "chars": sum(len(l) for l in lines)})

    # ---- 章节识别（行级粒度）----
    per_line_section, page_section, method = detect_sections(doc, pages_lines)
    log(f"   {n_pages} 页 | 表格 {n_tables} 张 {dict(table_methods)} | 章节方式: {method}")

    # ---- 展开成原子，按页内行序贴章节 ----
    atom_list: list[dict] = []
    for i, elems in enumerate(pages_elems):
        secs = per_line_section[i] or []
        fallback = page_section[i] if i < len(page_section) else UNKNOWN_SECTION
        for e in elems:
            ls = e.get("line_start", 0)
            sec = secs[ls] if ls < len(secs) else (secs[-1] if secs else fallback)
            if sec == UNKNOWN_SECTION:
                sec = fallback
            if e["kind"] == "text":
                atom_list.append({
                    "kind": "text", "page": i + 1, "text": e["text"], "section": sec,
                    "multicol": e.get("multicol", False),
                })
            else:
                hdr_rows, data_rows = split_header(e["rows"])
                if hdr_rows:
                    header = "\n".join(render_table(hdr_rows).splitlines())
                else:
                    header = e.get("caption", "")
                base = dict(page=i + 1, section=sec, table_id=e.get("table_id"), header=header)
                if hdr_rows:
                    atom_list.append({
                        **base, "kind": "table-row", "is_header": True, "row_index": 0,
                        "text": render_table(hdr_rows),
                    })
                for k, row in enumerate(data_rows):
                    if not any(c.strip() for c in row):
                        continue
                    atom_list.append({
                        **base, "kind": "table-row", "is_header": False, "row_index": k + 1,
                        "text": " | ".join(row),
                    })

    # ---- 切块 ----
    chunks = build_chunks(atom_list, code, name)

    # ---- 质检 ----
    qc = []
    for st in page_stats:                       # fitz 提取异常的页，如实记录
        qc.append({"page": st["page"], "chars": st["chars"], "reason": st["reason"]})
    failed_pages = {st["page"] for st in page_stats}
    for pm in pages_meta:
        page = pm["page"]
        if page in failed_pages:
            continue
        if pm["chars"] == 0:
            qc.append({"page": page, "chars": 0, "reason": "无文本层（疑似扫描图/纯图页）"})
        elif pm["chars"] < SUSPICIOUS_CHARS:
            qc.append({
                "page": page, "chars": pm["chars"],
                "reason": f"字数偏低（{pm['chars']} < {SUSPICIOUS_CHARS}），多为章节分隔页/图表页/目录页",
            })
    doc.close()

    sections = Counter(c["section"] for c in chunks)
    return {
        "code": code, "name": name, "file": filename,
        "total_pages": n_pages,
        "manifest_pages": manifest_pages,
        "extracted_pages": sum(1 for p in pages_meta if p["chars"] > 0),
        "chars": sum(p["chars"] for p in pages_meta),
        "tables": n_tables,
        "rotated_pages": rot_pages,
        "table_methods": dict(table_methods),
        "section_method": method,
        "chunks": chunks,
        "sections": dict(sections),
        "qc": qc,
    }


def build_chunks(atom_list: list[dict], code: str, name: str) -> list[dict]:
    """按原子边界做滑动窗口切块，目标 TARGET_TOKENS，重叠约 OVERLAP_TOKENS。"""
    if not atom_list:
        return []
    toks = [n_tokens(a["text"]) for a in atom_list]
    n = len(atom_list)
    chunks = []
    start = 0
    idx = 0
    while start < n:
        end, acc = start, 0
        while end < n and (acc < TARGET_TOKENS or end == start):
            acc += toks[end]
            end += 1
        piece = atom_list[start:end]

        # 组装文本；表格若从中间开始，补表头
        parts, seen_tables = [], set()
        for a in piece:
            if a["kind"] == "table-row":
                tid = a["table_id"]
                if tid not in seen_tables:
                    seen_tables.add(tid)
                    if not a["is_header"] and a["header"].strip():
                        parts.append(f"[表头] {a['header']}")
            parts.append(a["text"])
        text = "\n".join(p for p in parts if p.strip())

        idx += 1
        chunks.append({
            "chunk_id": f"{code}_{idx:04d}",
            "code": code,
            "name": name,
            "period": REPORT_PERIOD,
            "section": piece[0].get("section", UNKNOWN_SECTION),
            "page_start": min(a["page"] for a in piece),
            "page_end": max(a["page"] for a in piece),
            "chunk_index": idx,
            "tokens": n_tokens(text),
            "chars": len(text),
            "has_table": any(a["kind"] == "table-row" for a in piece),
            "has_multicol": any(a.get("multicol") for a in piece),
            "text": text,
        })

        if end >= n:
            break
        # 回退起点，使重叠 ≈ OVERLAP_TOKENS（只落在原子边界）
        tail, s = 0, end
        while s > start and tail + toks[s - 1] <= OVERLAP_TOKENS:
            s -= 1
            tail += toks[s]
        start = s if s > start else start + 1
    return chunks


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_manifest() -> dict[str, dict]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {r["code"]: r for r in data["reports"]}


def load_extracts() -> dict[str, dict]:
    """读回 extract/ 下已有的逐家结果（跳过已完成 + 最终合并都靠它）。"""
    out: dict[str, dict] = {}
    if not EXTRACT_DIR.exists():
        return out
    for p in sorted(EXTRACT_DIR.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log(f"! {p.name} 读取失败（{exc}），将重新提取")
    return out


def merge_chunks(results: dict[str, dict]) -> int:
    """把所有逐家结果合并成 chunks.jsonl，按代码排序，保证可重复。"""
    lines = []
    for code in sorted(results):
        for c in results[code]["chunks"]:
            lines.append(json.dumps(c, ensure_ascii=False))
    atomic_write_text(CHUNKS_PATH, "\n".join(lines) + "\n")
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", help="逗号分隔代码，默认全部")
    ap.add_argument("--sample", type=int, help="抽样自检：随机打印 N 块")
    ap.add_argument("--force", action="store_true", help="忽略 extract/ 已有结果，重新提取")
    ap.add_argument("--merge-only", action="store_true", help="跳过提取，只做合并与报告")
    args = ap.parse_args()

    man = load_manifest()
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else list(man)

    if args.sample:
        return sample_check(args.sample)

    EXTRACT_DIR.mkdir(exist_ok=True)
    results = load_extracts()
    t0 = time.time()

    for code in codes:
        m = man.get(code)
        if not m:
            log(f"! manifest 中无 {code}")
            continue
        if code in results and not args.force:
            r = results[code]
            log(f"{r['name']} | {r['total_pages']} 页 | {len(r['chunks'])} 块 | 已有结果，跳过")
            continue
        if args.merge_only:
            continue
        t1 = time.time()
        r = process_report(code, m["name"], m["filename"], m["pages"])
        r["seconds"] = round(time.time() - t1, 1)
        # 每份处理完立刻单独落盘（原子写），写完再处理下一份
        atomic_write_text(extract_path(code),
                          json.dumps(r, ensure_ascii=False, indent=1))
        results[code] = r
        log(f"{r['name']} | {r['total_pages']} 页 | {len(r['chunks'])} 块 | {r['seconds']:.1f}s")

    ordered = [results[c] for c in man if c in results]
    n_chunks = merge_chunks(results)
    write_report(ordered, time.time() - t0)

    log("\n" + "=" * 74)
    log(f"{'代码':<8}{'简称':<10}{'页数':>6}{'提取页':>7}{'字数':>10}{'表':>5}{'块数':>7}  章节方式")
    log("-" * 74)
    for r in ordered:
        log(f"{r['code']:<8}{r['name']:<10}{r['total_pages']:>6}{r['extracted_pages']:>7}"
            f"{r['chars']:>10}{r['tables']:>5}{len(r['chunks']):>7}  {r['section_method']}")
    log("-" * 74)
    log(f"总块数 {n_chunks} | 总字数 {sum(r['chars'] for r in ordered):,}"
        f" | 耗时 {time.time() - t0:.0f}s")
    log(f"\nchunks -> {CHUNKS_PATH}\n报告   -> {REPORT_PATH}")
    return 0


def sample_check(k: int) -> int:
    rows = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        log("chunks.jsonl 为空")
        return 1
    rng = random.Random(20260630)
    tables = [r for r in rows if r["has_table"]]
    multicols = [r for r in rows if r.get("has_multicol")]

    picked: list[dict] = []

    def take(pool: list[dict]) -> None:
        pool = [r for r in pool if r["chunk_id"] not in {p["chunk_id"] for p in picked}]
        if pool:
            picked.append(rng.choice(pool))

    take(tables)                                  # 至少 1 块表格块
    take(multicols)                               # 至少 1 块双栏页
    rest = [r for r in rows if r["chunk_id"] not in {p["chunk_id"] for p in picked}]
    picked += rng.sample(rest, min(k - len(picked), len(rest)))
    rng.shuffle(picked)

    for r in picked:
        log("=" * 78)
        log(f"{r['chunk_id']}  {r['name']}({r['code']})  {r['period']}  章节={r['section']}")
        log(f"页码 p{r['page_start']}-{r['page_end']}  块#{r['chunk_index']}  "
            f"tokens={r['tokens']} 字符={r['chars']}  "
            f"含表格={r['has_table']}  含双栏={r.get('has_multicol', False)}")
        log("-" * 78)
        log(r["text"])
    return 0


def write_report(results: list[dict], elapsed: float) -> None:
    L = []
    L.append("# PDF 提取与切块质检报告\n")
    L.append(f"生成时间：{datetime.now(BJT).isoformat(timespec='seconds')}  ")
    L.append(f"切块参数：目标 {TARGET_TOKENS} token，重叠 ≈{OVERLAP_TOKENS} token  \n")

    L.append("## 一、逐份提取概况\n")
    L.append("| 代码 | 简称 | 文件 | 总页数(manifest) | 成功提取页 | 提取字数 | 识别表格 | 块数 | 章节方式 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        mp = r["manifest_pages"]
        ok = "✅" if mp == r["total_pages"] else f"❌({mp})"
        L.append(f"| {r['code']} | {r['name']} | {r['file']} | {r['total_pages']} {ok} | "
                 f"{r['extracted_pages']} | {r['chars']:,} | {r['tables']} | "
                 f"{len(r['chunks'])} | {r['section_method']} |")
    L.append("")

    L.append("## 二、表格提取方式（级联统计）\n")
    L.append("| 代码 | 简称 | pdfplumber | 坐标重建 | pdfplumber(偏窄未重建) |")
    L.append("|---|---|---|---|---|")
    for r in results:
        tm = r["table_methods"]
        L.append(f"| {r['code']} | {r['name']} | {tm.get('pdfplumber', 0)} | "
                 f"{tm.get('rebuilt', 0)} | {tm.get('pdfplumber-narrow', 0)} |")
    L.append("")

    L.append("## 二·补、横排页（/Rotate）归一\n")
    rot = [r for r in results if r.get("rotated_pages")]
    if rot:
        L.append("fitz 的 `get_text` 不理会页面旋转，横排页给的是「内容坐标」，"
                 "文字在那里是竖排的（`dir=(0,-1)`），表格会碎成「一行一个数字」。"
                 "下列页已乘 `page.rotation_matrix` 映射回显示坐标：\n")
        L.append("| 代码 | 简称 | 横排页数 |")
        L.append("|---|---|---|")
        for r in rot:
            L.append(f"| {r['code']} | {r['name']} | {r['rotated_pages']} |")
    else:
        L.append("无。")
    L.append("")

    L.append("## 三、章节分布\n")
    L.append("| 代码 | 简称 | 各章节块数 |")
    L.append("|---|---|---|")
    for r in results:
        dist = "、".join(f"{k}:{v}" for k, v in sorted(r["sections"].items(), key=lambda x: -x[1]))
        L.append(f"| {r['code']} | {r['name']} | {dist} |")
    L.append("")

    L.append("## 四、可疑页清单\n")
    L.append(f"判定标准：单页提取字数 < {SUSPICIOUS_CHARS} 字，或完全无文本层。")
    L.append("均为如实记录，未做静默跳过。\n")
    any_qc = False
    for r in results:
        if not r["qc"]:
            continue
        any_qc = True
        L.append(f"### {r['code']} {r['name']}（{len(r['qc'])} 页）\n")
        L.append("| 页码 | 字数 | 原因 |")
        L.append("|---|---|---|")
        for q in r["qc"]:
            L.append(f"| p{q['page']} | {q['chars']} | {q['reason']} |")
        L.append("")
    if not any_qc:
        L.append("无。\n")

    L.append("## 五、页数核对\n")
    bad = [r for r in results if r["manifest_pages"] != r["total_pages"]]
    if bad:
        for r in bad:
            L.append(f"- ❌ {r['code']} {r['name']}：manifest {r['manifest_pages']} 页，实际 {r['total_pages']} 页")
    else:
        L.append("12 份全部与 manifest 记录一致，无缺页。")
    L.append("")

    L.append("## 六、已知局限\n")
    L.append("1. **农行的表格标记不完整**：农行表格多无框线，pdfplumber 只识别出 9 张，"
             "has_table 标记覆盖 10/193；内容通过正文通道保留，"
             "但按 has_table 过滤会漏。农行的表格类问题在检索评测中需单独观察。")
    L.append("2. **少数坐标重建的表仍有列错位**：883 张表中 370 张走坐标重建，多数可用，"
             "但跨页 / 合并单元格的复杂表重建后会出现数字与标签错行"
             "（实例：工行递延所得税变动表，块 `601398_0123`）。")
    L.append("")

    # 增量重跑时本次 elapsed≈0（全部命中 extract/ 缓存），逐份耗时之和才是有意义的数字
    secs = sum(r.get("seconds", 0) for r in results)
    if secs:
        L.append(f"\n_提取耗时合计 {secs:.0f} 秒（12 份逐份之和；命中缓存的份沿用上次记录）_\n")
    else:
        L.append(f"\n_总耗时 {elapsed:.0f} 秒_\n")

    atomic_write_text(REPORT_PATH, "\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
