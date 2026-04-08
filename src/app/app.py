import pandas as pd
import numpy as np
import os, json, joblib, hashlib
from flask import Flask, request, jsonify, send_from_directory

# ── Config ──
SERVICE_TYPE = os.getenv("SERVICE_TYPE", "analytics").lower()
SHARED_DIR = os.getenv("SHARED_DIR", "/app/shared")
PORT = int(os.getenv("PORT", 5000))

app = Flask(__name__, static_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static'))

def normalize_state(s):
    if not s: return ""
    s = str(s).upper().replace("&", " & ").replace("  ", " ").strip()
    if "DELHI" in s: return "Delhi"
    if "A & N" in s: return "A & N Islands"
    if "D & N" in s: return "D & N Haveli"
    if "DAMAN" in s: return "Daman & Diu"
    if "JAMMU" in s: return "Jammu and Kashmir"
    if "ODISHA" in s or "ORISSA" in s: return "Orissa"
    if "UTTARAKHAND" in s or "UTTARANCHAL" in s: return "Uttaranchal"
    m = {"UTTAR PRADESH": "Uttar Pradesh", "MADHYA PRADESH": "Madhya Pradesh", "ANDHRA PRADESH": "Andhra Pradesh", 
         "WEST BENGAL": "West Bengal", "HIMACHAL PRADESH": "Himachal Pradesh", "ARUNACHAL PRADESH": "Arunachal Pradesh",
         "JHARKHAND": "Jharkhand", "CHHATTISGARH": "Chhattisgarh"}
    return m.get(s, s.title())

# ── State Registry (Comprehensive) ──
STATE_META = {
    "Andhra Pradesh": [15.91, 79.74], "Arunachal Pradesh": [28.21, 94.72], "Assam": [26.20, 92.93],
    "Bihar": [25.09, 85.31], "Chhattisgarh": [21.27, 81.86], "Delhi": [28.61, 77.20], "Goa": [15.29, 74.12],
    "Gujarat": [22.25, 71.19], "Haryana": [29.05, 76.08], "Himachal Pradesh": [31.10, 77.17],
    "Jammu and Kashmir": [33.77, 76.57], "Jharkhand": [23.61, 85.27], "Karnataka": [15.31, 75.71],
    "Kerala": [10.85, 76.27], "Madhya Pradesh": [22.97, 78.65], "Maharashtra": [19.75, 75.71],
    "Manipur": [24.66, 93.90], "Meghalaya": [25.46, 91.36], "Mizoram": [23.16, 92.93], "Nagaland": [26.15, 94.56],
    "Orissa": [20.95, 85.09], "Punjab": [31.14, 75.34], "Rajasthan": [27.02, 74.21], "Sikkim": [27.53, 88.51],
    "Tamil Nadu": [11.12, 78.65], "Telangana": [18.11, 79.01], "Tripura": [23.94, 91.98],
    "Uttar Pradesh": [26.84, 80.94], "Uttaranchal": [30.06, 79.01], "West Bengal": [22.98, 87.85]
}

def normalize_district(name):
    if not name: return ""
    n = name.upper().replace(".", "").replace("-", " ").strip()
    if any(x in n for x in ["GRP", "RLY"]): return "G.R.P. (RAILWAY)"
    for unit in ["CAW", "EOW", "STF"]: 
        if unit in n: return f"{unit[0]}.{unit[1]}.{unit[2]}."
    return n

# ── Global Data Store ──
data = {"model": None, "scaler": None, "meta": None, "features": None, "risks": None, "spark": None}

def load_all():
    try:
        paths = {k: os.path.join(SHARED_DIR, f) for k, f in {"model": "model.pkl", "scaler": "scaler.pkl", "meta": "model_meta.joblib", "features": "cleaned_crime_features.csv", "risks": "district_risks.json"}.items()}
        if os.path.exists(paths["model"]): data["model"], data["scaler"], data["meta"] = joblib.load(paths["model"]), joblib.load(paths["scaler"]), joblib.load(paths["meta"])
        if os.path.exists(paths["features"]):
            df = pd.read_csv(paths["features"])
            df['STATE/UT'] = df['STATE/UT'].apply(normalize_state)
            df['DISTRICT'] = df['DISTRICT'].apply(normalize_district)
            data["features"] = df.groupby(['STATE/UT', 'DISTRICT']).sum().reset_index()
        if os.path.exists(paths["risks"]):
            with open(paths["risks"]) as f:
                raw_risks = json.load(f)
                data["risks"] = {f"{normalize_state(k.split('|')[0])}|{normalize_district(k.split('|')[1])}": v for k, v in raw_risks.items()}
        if SERVICE_TYPE == "prediction":
            from pyspark.sql import SparkSession
            data["spark"] = SparkSession.builder.appName("AE").master("local[*]").getOrCreate()
            p = "/app/data/Crimes_in_india_2001-2013.csv"
            if os.path.exists(p): data["spark"].read.csv(p, header=True).filter("UPPER(DISTRICT) NOT LIKE '%TOTAL%'").createOrReplaceTempView("crimes")
        return True
    except Exception as e: print(f"Load Error: {e}"); return False

load_all()

@app.route("/")
def index(): return send_from_directory(app.static_folder, f"{SERVICE_TYPE}.html")

@app.route("/api/hierarchy")
def get_hierarchy():
    if data["features"] is None: return jsonify({})
    return jsonify({k: sorted(list(v)) for k, v in data["features"].groupby('STATE/UT')['DISTRICT'].unique().to_dict().items()})

@app.route("/api/query", methods=["POST"])
def run_query():
    q = request.json.get("query", "")
    if not q.upper().strip().startswith("SELECT") or not data["spark"]: return jsonify({"error": "No engine"}), 403
    try: return jsonify(data["spark"].sql(q).limit(50).toPandas().to_dict(orient='records'))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/district-intensity")
def get_district_intensity():
    if data["features"] is None: return jsonify([])
    cols = data["meta"]['features'] if data["meta"] else [c for c in data["features"].columns if c not in ['STATE/UT', 'DISTRICT']]
    df = data["features"].copy()
    df['total'] = df[cols].sum(axis=1)
    return jsonify([{"district": r['DISTRICT'], "state": r['STATE/UT'], "total": int(r['total'])} for _, r in df.sort_values('total', ascending=False).head(20).iterrows()])

@app.route("/api/state-risk")
def get_state_risk():
    if not data["risks"]: return jsonify({})
    s_agg = {}
    if data["features"] is not None:
        cols = data["meta"]['features'] if data["meta"] else [c for c in data["features"].columns if c not in ['STATE/UT', 'DISTRICT']]
        df = data["features"].copy()
        df['total'] = df[cols].sum(axis=1)
        for s, row in df.groupby('STATE/UT').sum().iterrows():
            if s in STATE_META: s_agg[s] = {"total_crimes": int(row['total']), "features": row[cols].to_dict()}
    r_accum = {}
    for k, v in data["risks"].items():
        s = k.split('|')[0]
        if s in STATE_META:
            if s not in r_accum: r_accum[s] = []
            r_accum[s].append({"High Risk": 3, "Medium Risk": 2, "Low Risk": 1}.get(v['risk_level'], 0))
    res = {}
    for s, scrs in r_accum.items():
        avg = sum(scrs)/len(scrs) if scrs else 0
        hubs = ["Maharashtra", "Delhi", "Uttar Pradesh", "Madhya Pradesh", "Bihar"]
        if s in hubs or avg >= 1.5:
            lvl = "High Risk"
        elif avg >= 1.05:
            lvl = "Medium Risk"
        else:
            lvl = "Low Risk"
        res[s] = {**s_agg.get(s, {"total_crimes": 0, "features": {}}), "risk_score": {"High Risk": 3, "Medium Risk": 2, "Low Risk": 1}.get(lvl, 0), "risk_level": lvl}
    if "Telangana" not in res and "Andhra Pradesh" in res: res["Telangana"] = res["Andhra Pradesh"].copy()
    return jsonify(res)

@app.route("/api/predict-district")
def predict_district():
    s, d = normalize_state(request.args.get('state')), normalize_district(request.args.get('district'))
    key = f"{s}|{d}"
    if data["risks"] and key in data["risks"] and data["features"] is not None:
        row = data["features"][(data["features"]['STATE/UT']==s) & (data["features"]['DISTRICT']==d)]
        if not row.empty: return jsonify({"risk": data["risks"][key]["risk_level"], "features": {c: float(row[c].values[0]) for c in data["meta"]['features']}})
    return jsonify({"error": "Not found"}), 404

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": SERVICE_TYPE, "data_loaded": data["features"] is not None})

if __name__ == "__main__": app.run(host="0.0.0.0", port=PORT, debug=True)
