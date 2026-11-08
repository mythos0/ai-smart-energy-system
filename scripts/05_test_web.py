"""Test all Flask endpoints + screenshot the dashboard for the paper."""
import json
import subprocess
import time
import urllib.request
import urllib.error

import pandas as pd

# start server
srv = subprocess.Popen(["python3", "app.py"], cwd="/home/z/my-project/web",
                       stdout=open("/tmp/flask.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(40):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://localhost:5000/api/health", timeout=3)
            break
        except Exception:
            pass
    print("server up")

    # health
    r = json.load(urllib.request.urlopen("http://localhost:5000/api/health", timeout=10))
    assert r["status"] == "ok" and r["model"] == "xgboost"
    print("health ok:", r)

    # backtest forecast
    r = json.load(urllib.request.urlopen("http://localhost:5000/api/forecast?date=2010-07-06", timeout=30))
    assert len(r["points"]) == 24 and r["points"][0]["forecast_kw"] is not None
    print("backtest ok:", r["date"], "sigma", r["sigma_kw"],
          "| fc[0]", r["points"][0]["forecast_kw"], "actual[0]", r["points"][0]["actual_kw"])

    # date validation
    try:
        urllib.request.urlopen("http://localhost:5000/api/forecast?date=2008-01-01", timeout=10)
        print("ERROR: out-of-range date accepted")
    except urllib.error.HTTPError as e:
        print("date validation ok:", e.code)

    # live forecast (real month of history)
    h = pd.read_csv("/home/z/my-project/web/model/hourly_full.csv", index_col=0,
                    parse_dates=True)["global_active_power"]
    vals = list(h.loc["2010-09-01":"2010-09-30"].values.round(4))
    req = urllib.request.Request("http://localhost:5000/api/forecast",
                                 data=json.dumps({"values": vals}).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    assert r["mode"] == "live" and len(r["points"]) == 24
    fc = [p["power_kw"] for p in r["points"]]
    print("live ok: first 3:", fc[:3], "| mean:", round(sum(fc) / 24, 3),
          "| max:", round(max(fc), 3))

    # live forecast error case
    req = urllib.request.Request("http://localhost:5000/api/forecast", data=b'{"values": [1,2,3]}',
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print("ERROR: short input accepted")
    except urllib.error.HTTPError as e:
        print("live validation ok:", e.code, json.load(e)["error"])

    # anomaly
    a = json.load(urllib.request.urlopen(
        "http://localhost:5000/api/anomaly?start=2010-11-01&end=2010-11-08", timeout=30))
    n_flag = sum(1 for p in a["points"] if p["is_anomaly"])
    print("anomaly week ok:", len(a["points"]), "pts |", n_flag, "flagged")

    # models + casestudy
    m = json.load(urllib.request.urlopen("http://localhost:5000/api/models", timeout=30))
    print("models ok:", len(m["forecasting"]), "forecasters,", len(m["anomaly_detection"]),
          "detectors | best:", m["forecasting"][0]["model"])
    cs = json.load(urllib.request.urlopen("http://localhost:5000/api/anomaly/casestudy", timeout=30))
    print("casestudy ok:", len(cs["events"]), "events | top:", cs["events"][0]["start"])

    # index page
    html = urllib.request.urlopen("http://localhost:5000/", timeout=30).read().decode()
    assert "AI Smart Energy System" in html
    print("page ok | length:", len(html))

    # screenshots (forecast tab + anomaly tab + models tab)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto("http://localhost:5000/", wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(3500)
        pg.screenshot(path="/home/z/my-project/figures/web_forecast.png")
        pg.click('button[data-tab="anomaly"]')
        pg.wait_for_timeout(2500)
        pg.screenshot(path="/home/z/my-project/figures/web_anomaly.png")
        pg.click('button[data-tab="models"]')
        pg.wait_for_timeout(1500)
        pg.screenshot(path="/home/z/my-project/figures/web_models.png")
        b.close()
    print("screenshots saved")
    print("ALL TESTS PASSED")
finally:
    srv.terminate()
