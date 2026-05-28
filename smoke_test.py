"""
Headless smoke test for the Shiny app. Walks every tab, captures screenshots,
and reports any JS console errors or Python server-side rendering failures.
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8765"
OUT = Path(__file__).parent / "outputs" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)

TABS = ["Overview", "DoWhy", "EconML", "PyMC", "CausalNex", "Synthesis"]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 900})

        console_msgs, page_errors = [], []
        page.on("console", lambda msg: console_msgs.append((msg.type, msg.text)))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        await page.goto(URL, wait_until="networkidle")
        # First reactive render takes a moment
        await asyncio.sleep(8)
        await page.screenshot(path=str(OUT / "00_initial.png"), full_page=True)

        for i, tab in enumerate(TABS):
            try:
                await page.get_by_role("link", name=tab).click(timeout=5000)
            except Exception:
                # navbar items render as anchors; fall back to text selector
                await page.locator(f"a:has-text('{tab}')").first.click(timeout=5000)
            # let reactives settle (EconML refits can take ~5s)
            await asyncio.sleep(35)
            await page.screenshot(path=str(OUT / f"{i+1:02d}_{tab.lower()}.png"),
                                  full_page=True)
            print(f"  tab '{tab}': OK")

        errors = [m for m in console_msgs if m[0] in ("error",)]
        print(f"\nConsole errors: {len(errors)}")
        for level, text in errors[:10]:
            print(f"  [{level}] {text}")
        print(f"Page errors: {len(page_errors)}")
        for err in page_errors[:10]:
            print(f"  {err}")
        await browser.close()
        return len(errors) + len(page_errors) == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
