#!/usr/bin/env python3
"""第四步：给 Streamlit 页面拍截图（交作业用）。

    .venv/Scripts/python.exe -m streamlit run app.py     # 先起服务
    .venv/Scripts/python.exe shot.py                     # 再拍照

产物在 shots/。只用于出图，不参与运行时逻辑。
"""
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "shots"
URL = "http://localhost:8501"

Q1 = "招商银行 2026 年上半年净息差是多少？"
Q3 = "工商银行上半年手续费及佣金净收入是多少、同比变化如何？"
BASE_H = 1200

# (文件名, 问题, 是否全展开, 侧栏检索模式, 是否开公司限定, 是否开口径扩展)
QUESTIONS = [
    ("q1_招商银行净息差", Q1, False, None, True, True),
    ("q2_跨公司全景题", "2026 年上半年哪几家银行的营业收入同比增速最高？", False, None, True, True),
    ("q3_工商银行手续费", Q3, False, None, True, True),
    ("q3_全展开", Q3, True, None, True, True),
    # 同一问题关掉公司过滤、逐个模式跑：复现第三步 BM25 首条错公司，并看两路各自的表现
    ("q1_只用向量_无公司过滤", Q1, False, "只用向量", False, True),
    ("q1_只用BM25_无公司过滤", Q1, False, "只用 BM25", False, True),
    ("q1_混合_无公司过滤", Q1, False, "混合（向量 + BM25 + RRF）", False, True),
    # 口径扩展开/关：关掉应回到「材料里没有净息差」的拒答
    ("q1_关口径扩展", Q1, False, None, True, False),
]

FIRST_RENDER = 240_000      # 首次打开要等索引 + 向量模型加载
ANSWER_WAIT = 120_000       # 点完提问等答案（没配 key 时会很快转成提示）
MAX_H = 6000                # 截图高度上限，防止全展开时出一张几万像素的图


def content_height(page) -> int:
    """Streamlit 的滚动发生在内层容器里（[data-testid="stMain"]），full_page 截不到，
    只能自己量高度再把视口撑到那么高。"""
    return page.evaluate(
        "() => { const e = document.querySelector('[data-testid=\"stMain\"]');"
        " return e ? e.scrollHeight : document.body.scrollHeight; }")


def expand_all(page) -> None:
    heads = page.locator('[data-testid="stExpander"] summary')
    for i in range(heads.count()):
        h = heads.nth(i)
        if h.get_attribute("aria-expanded") != "true":
            h.click()
    page.wait_for_timeout(1200)


def set_check(page, label: str, want: bool) -> None:
    """按 aria-label 定位侧栏的 checkbox（现在有两个，按 role 取会撞）。

    <input> 被外层 label / svg 盖住，直接点会被判定「pointer events 被拦截」，
    所以状态读 input，点击点 label 文字。

    定位必须限定 role=checkbox：get_by_label 会连旁边的「?」帮助按钮一起命中
    （它的 aria-label 是 "Help for 启用公司限定检索"），触发 strict mode 报错。
    """
    if page.get_by_role("checkbox", name=label).is_checked() == want:
        return
    page.get_by_text(label, exact=True).click()
    page.wait_for_timeout(800)


def shoot(page, name: str, question: str, open_all: bool, first: bool, company: bool,
          mode: str | None = None, expand: bool = True) -> None:
    # 每次先缩回基准高度：stMain 的 scrollHeight 至少等于视口高度，
    # 上一张撑到 6000 后不缩回来，这一张就会量出 6000，一路虚高下去
    page.set_viewport_size({"width": 1500, "height": BASE_H})
    if first:
        page.wait_for_selector('input[type="text"]', timeout=FIRST_RENDER)
    if mode:
        page.get_by_text(mode, exact=True).click()
        page.wait_for_timeout(800)
    set_check(page, "启用公司限定检索", company)
    set_check(page, "启用口径/同义词扩展", expand)
    box = page.locator('input[type="text"]').first
    box.click()
    box.press("Control+a")
    box.fill(question)
    page.get_by_role("button", name="提问").click()
    page.wait_for_selector("text=/召回的 \\d+ 个块/", timeout=ANSWER_WAIT)
    page.wait_for_timeout(1500)          # 等展开动画和 markdown 排版落定
    if open_all:
        expand_all(page)
    h = min(max(content_height(page), 1000), MAX_H)
    page.set_viewport_size({"width": 1500, "height": h})
    page.wait_for_timeout(800)
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path))
    print(f"  {path.relative_to(ROOT)}  ({1500}x{h})")


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        # 本机已装 Edge，直接用系统内核，省掉 195MB 的 chromium 下载
        # （`python -m playwright install chromium` 走镜像也卡在 0%）
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1500, "height": 1200},
                                device_scale_factor=1.2)   # 放大一点，截图里小字能看清
        print(f"打开 {URL} …")
        page.goto(URL, wait_until="domcontentloaded")
        for i, (name, q, open_all, mode, company, expand) in enumerate(QUESTIONS):
            tags = []
            if open_all:
                tags.append("全展开")
            if mode:
                tags.append(mode)
            tags.append(f"公司过滤{'开' if company else '关'}")
            tags.append(f"口径扩展{'开' if expand else '关'}")
            print(f"提问：{q}（{'，'.join(tags)}）")
            shoot(page, name, q, open_all, first=(i == 0), company=company,
                  mode=mode, expand=expand)
        browser.close()
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
