"""Times the hosted dashboard from a fresh browser: wakes it if suspended, uploads a defective and a normal part."""
import json, time, datetime
from playwright.sync_api import sync_playwright
R = "data/mvtec_ad_2/sheet_metal/test_public/"
OUT = "logs/"  # screenshots show dataset images, so they stay local
log = {"start_utc": datetime.datetime.utcnow().isoformat()}
def app_frame(page):
    return next((f for f in page.frames if f.url.endswith("/~/+/")), None)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_page(viewport={"width": 1400, "height": 1500})
    t0 = time.time(); page.goto("https://pm-inspection.streamlit.app/", timeout=180000)
    page.wait_for_timeout(5000)
    page.screenshot(path=OUT + "hosted_0_landing.png")
    log["landing_text"] = page.inner_text("body")[:400]
    btn = page.get_by_role("button", name="Yes, get this app back up!")
    log["asleep"] = btn.count() > 0
    if log["asleep"]:
        tw = time.time(); btn.click(); log["wake_clicked_s"] = round(tw - t0, 1)
    else:
        tw = t0
    fr = None
    for i in range(600):
        fr = app_frame(page)
        if fr and fr.locator("text=Visual inspection").count(): break
        if i % 30 == 0: page.screenshot(path=OUT + f"cold_wait_{i:03d}.png")
        page.wait_for_timeout(1000)
    log["app_ready_after_wake_s"] = round(time.time() - tw, 1)
    page.screenshot(path=OUT + "hosted_1_ready.png")
    for name, f in [("defect", "bad/004_regular.png"), ("good", "good/000_regular.png")]:
        t1 = time.time()
        fr.locator('input[type="file"]').set_input_files(R + f)
        fr.wait_for_selector(".lamp.ok, .lamp.ng", timeout=600000)
        log[name + "_decision_wall_s"] = round(time.time() - t1, 1)
        fr.wait_for_timeout(3000)
        txt = fr.locator("body").inner_text()
        log[name + "_text"] = [l for l in txt.splitlines() if any(k in l for k in ("ms", "MiB", "threshold", "loaded", "PASS", "REJECT"))][:12]
        page.screenshot(path=OUT + f"cold_2_{name}.png", full_page=True)
    b.close()
log["end_utc"] = datetime.datetime.utcnow().isoformat()
json.dump(log, open(OUT + "hosted_log.json", "w"), indent=1)
print(json.dumps(log, indent=1))
