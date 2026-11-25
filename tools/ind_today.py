#!/usr/bin/env python3
"""Daily country indicators for the Eklektik World country card.

Fetches the most recent non-empty value per country from the World Bank API
(population, men, women, life expectancy, GDP nominal and per capita, mobile
subscriptions and per 100, surface area) and ITU DataHub via World Bank Data360
(individuals using a computer, households with a computer).
Writes a/ind.json. Values carry their real data year. The file is only rewritten
when a value changed (the "date" field alone does not trigger a commit).
"""
import json, os, sys, time, urllib.request, urllib.parse, datetime

UA = {"User-Agent": "EklektikWorld/1.0 (https://eklektikchange-a11y.github.io/)"}
WB = {
    "pop": "SP.POP.TOTL", "m": "SP.POP.TOTL.MA.IN", "f": "SP.POP.TOTL.FE.IN",
    "life": "SP.DYN.LE00.IN", "gdpT": "NY.GDP.MKTP.CD", "gdp": "NY.GDP.PCAP.CD",
    "mob": "IT.CEL.SETS", "mob100": "IT.CEL.SETS.P2", "area": "AG.SRF.TOTL.K2",
}
D360 = {"pcu": "IT_INV_COMP", "pch": "IT_HH_COMP"}
# app code -> World Bank / ITU code
ALIAS = {"PSX": "PSE", "SDS": "SSD", "KOS": "XKX"}


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa
            err = e
            time.sleep(4 * (k + 1))
    raise RuntimeError("fetch failed %s: %s" % (url, err))


def wb(ind):
    d = get("https://api.worldbank.org/v2/country/all/indicator/%s?format=json&per_page=20000&mrnev=1" % ind)
    out = {}
    for x in d[1] or []:
        c = x.get("countryiso3code")
        if c and x.get("value") is not None:
            out[c] = [x["value"], int(x["date"])]
    return out, d[0].get("lastupdated")


def d360(ind):
    out, skip = {}, 0
    while True:
        d = get("https://data360api.worldbank.org/data360/data?" + urllib.parse.urlencode(
            {"DATABASE_ID": "ITU_DH", "INDICATOR": ind, "skip": skip}))
        v = d.get("value") or []
        for x in v:
            if x.get("SEX") not in (None, "_Z", "_T") or x.get("AGE") not in (None, "_Z", "_T") or x.get("URBANISATION") not in (None, "_Z", "_T"):
                continue
            try:
                y, val = int(x["TIME_PERIOD"]), float(x["OBS_VALUE"])
            except Exception:
                continue
            c = x["REF_AREA"]
            if c not in out or y > out[c][1]:
                out[c] = [round(val, 1), y]
        skip += len(v)
        if not v or skip >= d.get("count", 0):
            break
    return out


def main():
    outdir = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "a"
    path = os.path.join(outdir, "ind.json")
    data, upd = {}, {}
    for k, ind in WB.items():
        vals, lu = wb(ind)
        upd[ind] = lu
        for c, v in vals.items():
            if k in ("pop", "m", "f", "mob", "area"):
                v = [int(round(v[0])), v[1]]
            elif k in ("gdpT",):
                v = [int(round(v[0])), v[1]]
            elif k in ("gdp",):
                v = [round(v[0], 1), v[1]]
            else:
                v = [round(v[0], 2), v[1]]
            data.setdefault(c, {})[k] = v
    for k, ind in D360.items():
        try:
            for c, v in d360(ind).items():
                data.setdefault(c, {})[k] = v
        except Exception as e:
            print("ITU via Data360 skipped:", e, file=sys.stderr)
    for app, src in ALIAS.items():
        if src in data:
            data[app] = data[src]
    # drop aggregates (WB aggregate codes are not real countries but harmless); keep only 3-letter codes
    agg = set()
    try:
        cl = get("https://api.worldbank.org/v2/country?format=json&per_page=400")[1]
        agg = {x["id"] for x in cl if x["region"]["value"] == "Aggregates"}
    except Exception:
        pass
    data = {c: v for c, v in sorted(data.items()) if len(c) == 3 and c.isalpha() and c not in agg}
    old = None
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            old = None
    if old and old.get("c") == data and "--force" not in sys.argv:
        print("unchanged")
        return
    today = datetime.datetime.now(datetime.timezone.utc)
    j = {"date": today.strftime("%Y-%m-%d"), "fetched_at": today.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "src": {"wb": "World Bank WDI API", "itu": "ITU DataHub (World Bank Data360)", "wb_lastupdated": upd},
         "c": data}
    with open(path, "w") as f:
        json.dump(j, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("written", path, len(data))


if __name__ == "__main__":
    main()
