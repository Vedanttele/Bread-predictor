from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1600})
    page.goto("http://localhost:8506", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("text=Roti Predictor", timeout=15000)
    page.wait_for_timeout(800)
    page.get_by_role("button", name="Predict my targets").click()
    page.wait_for_selector("text=Roti count", timeout=15000)
    page.wait_for_timeout(800)
    page.get_by_role("button", name="Get recipe recommendation").click()
    page.wait_for_selector("text=No AI used", timeout=15000)
    page.wait_for_selector("text=Prep time", timeout=15000)
    page.wait_for_timeout(1200)
    page.screenshot(path="docs/screenshots/today_tab_no_llm_fallback.png", full_page=True)
    print("saved")
