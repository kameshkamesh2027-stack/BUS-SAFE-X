"""
BUS-SAFE X - AI-Powered Intelligent Bus Accident Response & Rescue Intelligence System
========================================================================================
Backend: Python + Flask
Database: SQLite (bussafe.db)
AI/ML: Rule-based intelligent accident detection (simulation-based AI detection)
GPS: Simulated GPS with nearest-hospital calculation
"""

import sqlite3
import json
import math
import time
import threading
import os
from datetime import datetime
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BUS_ID = "BUS-SAFE-X-001"
DB_PATH = os.path.join(os.path.dirname(__file__), "bussafe.db")
ACCIDENT_CONFIDENCE_THRESHOLD = 70  # percent

# Sensor weights used by the AI detection engine.
# Each entry: field name in state, maximum raw value, contribution weight (out of 100).
# Weights must sum to 100. Changing a weight here affects both detection and /api/aiml.
SENSOR_WEIGHTS = {
    "impact":       {"field": "sensor_impact",       "max": 10.0, "weight": 35},
    "damage":       {"field": "sensor_damage",       "max": 10.0, "weight": 25},
    "deceleration": {"field": "sensor_deceleration", "max": 10.0, "weight": 20},
    "fire":         {"field": "sensor_fire",         "max":  2.0, "weight": 10},
    "smoke":        {"field": "sensor_smoke",        "max":  2.0, "weight": 10},
}

# ---------------------------------------------------------------------------
# HOSPITAL DATASET (offline fallback - no external API required)
# Based on Tirupur / Coimbatore area, Tamil Nadu, India
# ---------------------------------------------------------------------------

HOSPITALS = [
    {"name": "Tirupur Government Hospital",      "lat": 11.1085, "lon": 77.3411, "phone": "0421-2200100"},
    {"name": "PSG Hospitals Coimbatore",         "lat": 11.0168, "lon": 76.9558, "phone": "0422-4345678"},
    {"name": "Kovai Medical Center",             "lat": 11.0201, "lon": 76.9641, "phone": "0422-4323800"},
    {"name": "Tirupur Apollo Clinic",            "lat": 11.1091, "lon": 77.3447, "phone": "0421-4262626"},
    {"name": "Sri Ramakrishna Hospital",         "lat": 11.0059, "lon": 76.9700, "phone": "0422-4500000"},
    {"name": "Erode Government Hospital",        "lat": 11.3410, "lon": 77.7172, "phone": "0424-2255111"},
    {"name": "Annur Primary Health Centre",      "lat": 11.2380, "lon": 77.1010, "phone": "0422-2677100"},
    {"name": "Dharapuram Government Hospital",   "lat": 10.7370, "lon": 77.5130, "phone": "04258-222030"},
]

# ---------------------------------------------------------------------------
# IN-MEMORY SYSTEM STATE
# ---------------------------------------------------------------------------

state = {
    "bus_id": BUS_ID,
    "timestamp": datetime.now().isoformat(),

    # GPS
    "gps_lat": 11.0510,
    "gps_lon": 77.0180,

    # Passengers
    "passengers_total": 32,
    "passengers_front": 12,
    "passengers_middle": 13,
    "passengers_rear": 7,

    # Sensors (raw values)
    "sensor_front_passengers": 12,
    "sensor_rear_passengers": 7,
    "sensor_impact": 0.5,          # 0-10 scale
    "sensor_damage": 0.3,          # 0-10 scale
    "sensor_deceleration": 0.2,    # 0-10 scale (sudden decel)
    "sensor_fire": 0,              # 0=none, 1=warning, 2=detected
    "sensor_smoke": 0,             # 0=none, 1=warning, 2=detected

    # AI / ML Detection
    "accident_confidence": 0.0,
    "accident_detected": False,
    "affected_zone": "None",

    # Fire / Smoke status
    "fire_status": "NORMAL",
    "smoke_status": "NORMAL",

    # Emergency
    "emergency_active": False,
    "emergency_status": "Standby",

    # Hospital
    "nearest_hospital": "Calculating...",
    "hospital_distance_km": 0.0,
    "hospital_lat": 0.0,
    "hospital_lon": 0.0,
    "hospital_phone": "",
}

# ---------------------------------------------------------------------------
# THREAD SAFETY
# ---------------------------------------------------------------------------

state_lock = threading.Lock()
"""
Single lock that guards all reads and writes of the ``state`` dict.

The background auto_save_loop thread and Flask request threads can
mutate state concurrently.  Every function that reads or writes state
must hold this lock so no thread sees a half-written snapshot.
"""

# ---------------------------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------------------------

def init_db():
    """Create SQLite tables if they do not exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY,
                    bus_id TEXT,
                    timestamp TEXT,
                    gps_lat REAL,
                    gps_lon REAL,
                    passengers_total INTEGER,
                    passengers_front INTEGER,
                    passengers_middle INTEGER,
                    passengers_rear INTEGER,
                    sensor_impact REAL,
                    sensor_damage REAL,
                    sensor_deceleration REAL,
                    sensor_fire INTEGER,
                    sensor_smoke INTEGER,
                    accident_confidence REAL,
                    accident_detected INTEGER,
                    fire_status TEXT,
                    smoke_status TEXT,
                    emergency_active INTEGER,
                    emergency_status TEXT,
                    nearest_hospital TEXT,
                    hospital_distance_km REAL,
                    affected_zone TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    description TEXT,
                    confidence REAL
                )
            """)
    except sqlite3.Error as e:
        print(f"[init_db] DB error: {e}", flush=True)


def save_state_to_db():
    """Persist current in-memory state to SQLite (upsert row id=1)."""
    with state_lock:
        snapshot = {**state,
                    "accident_detected": int(state["accident_detected"]),
                    "emergency_active":  int(state["emergency_active"])}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO system_state (
                    id, bus_id, timestamp, gps_lat, gps_lon,
                    passengers_total, passengers_front, passengers_middle, passengers_rear,
                    sensor_impact, sensor_damage, sensor_deceleration,
                    sensor_fire, sensor_smoke,
                    accident_confidence, accident_detected,
                    fire_status, smoke_status,
                    emergency_active, emergency_status,
                    nearest_hospital, hospital_distance_km, affected_zone
                ) VALUES (
                    1, :bus_id, :timestamp, :gps_lat, :gps_lon,
                    :passengers_total, :passengers_front, :passengers_middle, :passengers_rear,
                    :sensor_impact, :sensor_damage, :sensor_deceleration,
                    :sensor_fire, :sensor_smoke,
                    :accident_confidence, :accident_detected,
                    :fire_status, :smoke_status,
                    :emergency_active, :emergency_status,
                    :nearest_hospital, :hospital_distance_km, :affected_zone
                )
            """, snapshot)
    except sqlite3.Error as e:
        print(f"[save_state_to_db] DB error: {e}", flush=True)


def load_state_from_db():
    """Restore state from database on startup."""
    global state
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM system_state WHERE id=1")
        row = c.fetchone()
        conn.close()
        if row:
            restored = dict(row)
            restored.pop("id", None)
            restored["accident_detected"] = bool(restored["accident_detected"])
            restored["emergency_active"] = bool(restored["emergency_active"])
            with state_lock:
                state.update(restored)
    except sqlite3.Error as e:
        print(f"[load_state_from_db] DB error: {e}", flush=True)


def log_event(event_type, description, confidence=0.0):
    """Append an event to the event_log table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO event_log (timestamp, event_type, description, confidence) VALUES (?,?,?,?)",
                (datetime.now().isoformat(), event_type, description, confidence)
            )
    except sqlite3.Error as e:
        print(f"[log_event] DB error: {e}", flush=True)

# ---------------------------------------------------------------------------
# HAVERSINE DISTANCE (km)
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_hospital(lat, lon):
    """Return the nearest hospital dict with distance_km added."""
    nearest = None
    min_dist = float("inf")
    for h in HOSPITALS:
        d = haversine(lat, lon, h["lat"], h["lon"])
        if d < min_dist:
            min_dist = d
            nearest = h
    result = dict(nearest)
    result["distance_km"] = round(min_dist, 2)
    return result

# ---------------------------------------------------------------------------
# AI / ML ACCIDENT DETECTION ENGINE
# (Simulation-based AI detection — rule-based intelligent logic)
# Can be replaced with a trained ML model by swapping this function.
# ---------------------------------------------------------------------------

def run_ai_detection():
    """
    Compute accident confidence from sensor inputs.

    Inputs:
        sensor_impact       : 0-10  (physical impact force)
        sensor_damage       : 0-10  (structural damage level)
        sensor_deceleration : 0-10  (sudden deceleration magnitude)
        sensor_fire         : 0-2   (0=none, 1=warning, 2=detected)
        sensor_smoke        : 0-2   (0=none, 1=warning, 2=detected)

    Logic:
        Each sensor contributes a weighted score.
        Total weighted score is normalized to 0-100% confidence.

    Weights (out of 100):
        impact       : 35
        damage       : 25
        deceleration : 20
        fire         : 10
        smoke        : 10
    """
    with state_lock:
        confidence = round(
            sum((state[w["field"]] / w["max"]) * w["weight"]
                for w in SENSOR_WEIGHTS.values()),
            2
        )
        confidence = min(confidence, 100.0)

        state["accident_confidence"] = confidence
        detected = confidence >= ACCIDENT_CONFIDENCE_THRESHOLD
        state["accident_detected"] = detected

        # Fire / smoke status
        fire_map  = {0: "NORMAL", 1: "WARNING", 2: "DETECTED"}
        state["fire_status"]  = fire_map.get(state["sensor_fire"],  "NORMAL")
        state["smoke_status"] = fire_map.get(state["sensor_smoke"], "NORMAL")

        # Identify affected zone (simplified rule)
        if detected:
            fi = state["sensor_impact"]
            if fi >= 7:
                if state["sensor_front_passengers"] > state["sensor_rear_passengers"]:
                    state["affected_zone"] = "Front Zone"
                else:
                    state["affected_zone"] = "Rear Zone"
            else:
                state["affected_zone"] = "Multiple Zones"
            if not state["emergency_active"]:
                state["emergency_active"] = True
                state["emergency_status"] = "Emergency response activated"
                need_log = True
            else:
                need_log = False
        else:
            state["affected_zone"] = "None"
            if state["emergency_active"] is False:
                state["emergency_status"] = "Standby"
            need_log = False

        # Read GPS for hospital lookup before releasing the lock
        gps_lat = state["gps_lat"]
        gps_lon = state["gps_lon"]

    # Hospital lookup is pure computation — no state writes yet, lock not held
    h = find_nearest_hospital(gps_lat, gps_lon)

    with state_lock:
        state["nearest_hospital"]     = h["name"]
        state["hospital_distance_km"] = h["distance_km"]
        state["hospital_lat"]         = h["lat"]
        state["hospital_lon"]         = h["lon"]
        state["hospital_phone"]       = h.get("phone", "")
        state["timestamp"]            = datetime.now().isoformat()

    # log_event opens its own DB connection — call outside the lock
    if need_log:
        log_event("ACCIDENT_DETECTED", "Accident detected by AI engine", confidence)

# ---------------------------------------------------------------------------
# BACKGROUND AUTO-SAVE THREAD
# ---------------------------------------------------------------------------

# Small GPS increments that simulate bus movement along a road corridor.
# Each step advances lat/lon slightly so the frontend can compute real speed.
# One step every 10 s at ~0.00015 deg lat ≈ ~16 m  →  ~5.8 km/h base speed.
# Steps cycle through a simple back-and-forth path so the bus stays on-screen.
_GPS_STEPS = [
    ( 0.00015,  0.00010),
    ( 0.00012,  0.00015),
    ( 0.00008,  0.00018),
    ( 0.00005,  0.00012),
    ( 0.00000,  0.00008),
    (-0.00005,  0.00004),
    (-0.00008,  0.00000),
    (-0.00010, -0.00005),
    (-0.00012, -0.00010),
    (-0.00010, -0.00015),
    (-0.00008, -0.00012),
    (-0.00005, -0.00008),
    (-0.00002, -0.00004),
    ( 0.00003,  0.00000),
    ( 0.00008,  0.00005),
    ( 0.00012,  0.00008),
]
_gps_step_idx = 0


def auto_save_loop():
    """
    Background loop: runs every 10 seconds.
    - Advances simulated GPS position by one small step so the frontend
      always has a genuine position delta to compute speed from.
    - Runs AI detection on the new position.
    - Persists state to SQLite.
    GPS movement is suppressed while an emergency is active so the bus
    appears to have stopped at the accident site.
    """
    global _gps_step_idx
    while True:
        time.sleep(10)
        try:
            with state_lock:
                emergency = state["emergency_active"]

            # Only move the bus when there is no active emergency
            if not emergency:
                dlat, dlon = _GPS_STEPS[_gps_step_idx % len(_GPS_STEPS)]
                _gps_step_idx += 1
                with state_lock:
                    new_lat = state["gps_lat"] + dlat
                    new_lon = state["gps_lon"] + dlon
                    # Keep coordinates within Tamil Nadu bounding box
                    new_lat = max(10.5, min(13.5, new_lat))
                    new_lon = max(76.5, min(78.5, new_lon))
                    state["gps_lat"] = new_lat
                    state["gps_lon"] = new_lon

            run_ai_detection()
            save_state_to_db()
        except Exception as e:
            print(f"[auto_save_loop] Error: {e}", flush=True)

# ---------------------------------------------------------------------------
# FLASK ROUTES — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------------------------------------
# FLASK API — Dashboard
# ---------------------------------------------------------------------------

@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    run_ai_detection()
    return jsonify({
        "bus_id":               state["bus_id"],
        "timestamp":            state["timestamp"],
        "system_status":        "ACCIDENT DETECTED" if state["accident_detected"] else "NORMAL",
        "live":                 True,
        "gps_lat":              state["gps_lat"],
        "gps_lon":              state["gps_lon"],
        "passengers_total":     state["passengers_total"],
        "passengers_front":     state["passengers_front"],
        "passengers_middle":    state["passengers_middle"],
        "passengers_rear":      state["passengers_rear"],
        "accident_detected":    state["accident_detected"],
        "accident_confidence":  state["accident_confidence"],
        "fire_status":          state["fire_status"],
        "smoke_status":         state["smoke_status"],
        "nearest_hospital":     state["nearest_hospital"],
        "hospital_distance_km": state["hospital_distance_km"],
        "emergency_status":     state["emergency_status"],
        "emergency_active":     state["emergency_active"],
        "affected_zone":        state["affected_zone"],
    })

# ---------------------------------------------------------------------------
# FLASK API — Passengers
# ---------------------------------------------------------------------------

@app.route("/api/passengers", methods=["GET"])
def api_passengers():
    return jsonify({
        "total":  state["passengers_total"],
        "front":  state["passengers_front"],
        "middle": state["passengers_middle"],
        "rear":   state["passengers_rear"],
        "sensor_front": state["sensor_front_passengers"],
        "sensor_rear":  state["sensor_rear_passengers"],
    })

@app.route("/api/passengers/update", methods=["POST"])
def api_passengers_update():
    """Update individual zone counts and recalculate total."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    with state_lock:
        front  = int(data.get("front",  state["passengers_front"]))
        middle = int(data.get("middle", state["passengers_middle"]))
        rear   = int(data.get("rear",   state["passengers_rear"]))
        front  = max(0, front)
        middle = max(0, middle)
        rear   = max(0, rear)
        state["passengers_front"]          = front
        state["passengers_middle"]         = middle
        state["passengers_rear"]           = rear
        state["passengers_total"]          = front + middle + rear
        state["sensor_front_passengers"]   = front
        state["sensor_rear_passengers"]    = rear
        total = state["passengers_total"]
    save_state_to_db()
    return jsonify({"success": True, "total": total})

# ---------------------------------------------------------------------------
# FLASK API — Sensors
# ---------------------------------------------------------------------------

@app.route("/api/sensors", methods=["GET"])
def api_sensors():
    return jsonify({
        "sensor_front_passengers": state["sensor_front_passengers"],
        "sensor_rear_passengers":  state["sensor_rear_passengers"],
        "sensor_impact":           state["sensor_impact"],
        "sensor_damage":           state["sensor_damage"],
        "sensor_deceleration":     state["sensor_deceleration"],
        "sensor_fire":             state["sensor_fire"],
        "sensor_smoke":            state["sensor_smoke"],
        "fire_status":             state["fire_status"],
        "smoke_status":            state["smoke_status"],
    })

@app.route("/api/sensors/update", methods=["POST"])
def api_sensors_update():
    """Update one or more sensor values."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    float_fields = ["sensor_impact", "sensor_damage", "sensor_deceleration"]
    int_fields   = ["sensor_fire", "sensor_smoke",
                    "sensor_front_passengers", "sensor_rear_passengers"]
    int_clamps = {
        "sensor_fire":             (0, 2),
        "sensor_smoke":            (0, 2),
        "sensor_front_passengers": (0, 200),
        "sensor_rear_passengers":  (0, 200),
    }
    with state_lock:
        for f in float_fields:
            if f in data:
                state[f] = max(0.0, min(10.0, float(data[f])))
        for f in int_fields:
            if f in data:
                lo, hi = int_clamps[f]
                state[f] = max(lo, min(hi, int(data[f])))
    run_ai_detection()
    save_state_to_db()
    with state_lock:
        confidence = state["accident_confidence"]
    return jsonify({"success": True, "accident_confidence": confidence})

# ---------------------------------------------------------------------------
# FLASK API — AI / ML
# ---------------------------------------------------------------------------

@app.route("/api/aiml", methods=["GET"])
def api_aiml():
    run_ai_detection()
    return jsonify({
        "model_type":           "Simulation-based AI Detection (Rule-Based Weighted Scoring)",
        "threshold":            ACCIDENT_CONFIDENCE_THRESHOLD,
        "inputs": {
            "impact":           state["sensor_impact"],
            "damage":           state["sensor_damage"],
            "deceleration":     state["sensor_deceleration"],
            "fire":             state["sensor_fire"],
            "smoke":            state["sensor_smoke"],
        },
        "weights": {k: v["weight"] for k, v in SENSOR_WEIGHTS.items()},
        "scores": {
            f"{k}_score": round((state[v["field"]] / v["max"]) * v["weight"], 2)
            for k, v in SENSOR_WEIGHTS.items()
        },
        "accident_confidence":  state["accident_confidence"],
        "accident_detected":    state["accident_detected"],
        "detection_status":     "ACCIDENT DETECTED" if state["accident_detected"] else "NORMAL",
    })

# ---------------------------------------------------------------------------
# FLASK API — Emergency
# ---------------------------------------------------------------------------

@app.route("/api/emergency", methods=["GET"])
def api_emergency():
    run_ai_detection()
    h = find_nearest_hospital(state["gps_lat"], state["gps_lon"])
    return jsonify({
        "bus_id":               state["bus_id"],
        "timestamp":            state["timestamp"],
        "gps_lat":              state["gps_lat"],
        "gps_lon":              state["gps_lon"],
        "passengers_total":     state["passengers_total"],
        "passengers_front":     state["passengers_front"],
        "passengers_middle":    state["passengers_middle"],
        "passengers_rear":      state["passengers_rear"],
        "accident_confidence":  state["accident_confidence"],
        "accident_detected":    state["accident_detected"],
        "affected_zone":        state["affected_zone"],
        "fire_status":          state["fire_status"],
        "smoke_status":         state["smoke_status"],
        "emergency_active":     state["emergency_active"],
        "emergency_status":     state["emergency_status"],
        "nearest_hospital":     h["name"],
        "hospital_distance_km": h["distance_km"],
        "hospital_lat":         h["lat"],
        "hospital_lon":         h["lon"],
        "hospital_phone":       h.get("phone", ""),
        "sensor_impact":        state["sensor_impact"],
        "sensor_damage":        state["sensor_damage"],
        "sensor_deceleration":  state["sensor_deceleration"],
    })

@app.route("/api/emergency/reset", methods=["POST"])
def api_emergency_reset():
    """Reset emergency state (back to normal for demo re-use)."""
    with state_lock:
        state["accident_detected"]   = False
        state["accident_confidence"] = 0.0
        state["emergency_active"]    = False
        state["emergency_status"]    = "Standby"
        state["affected_zone"]       = "None"
        state["sensor_impact"]       = 0.5
        state["sensor_damage"]       = 0.3
        state["sensor_deceleration"] = 0.2
        state["sensor_fire"]         = 0
        state["sensor_smoke"]        = 0
        state["fire_status"]         = "NORMAL"
        state["smoke_status"]        = "NORMAL"
    run_ai_detection()
    save_state_to_db()
    log_event("RESET", "System reset to normal state", 0.0)
    return jsonify({"success": True, "message": "System reset to normal state."})

# ---------------------------------------------------------------------------
# FLASK API — Hospital / GPS
# ---------------------------------------------------------------------------

@app.route("/api/hospital", methods=["GET"])
def api_hospital():
    h = find_nearest_hospital(state["gps_lat"], state["gps_lon"])
    all_hospitals = []
    for hosp in HOSPITALS:
        d = haversine(state["gps_lat"], state["gps_lon"], hosp["lat"], hosp["lon"])
        entry = dict(hosp)
        entry["distance_km"] = round(d, 2)
        all_hospitals.append(entry)
    all_hospitals.sort(key=lambda x: x["distance_km"])
    return jsonify({
        "current_gps": {"lat": state["gps_lat"], "lon": state["gps_lon"]},
        "nearest":     h,
        "all":         all_hospitals,
    })

@app.route("/api/gps", methods=["GET"])
def api_gps():
    return jsonify({
        "lat":   state["gps_lat"],
        "lon":   state["gps_lon"],
        "label": f"{state['gps_lat']:.6f}, {state['gps_lon']:.6f}",
    })

@app.route("/api/gps/update", methods=["POST"])
def api_gps_update():
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    lat_val = float(data["lat"]) if "lat" in data else None
    lon_val = float(data["lon"]) if "lon" in data else None
    if lat_val is not None and not (-90.0 <= lat_val <= 90.0):
        return jsonify({"error": "lat must be between -90 and 90"}), 400
    if lon_val is not None and not (-180.0 <= lon_val <= 180.0):
        return jsonify({"error": "lon must be between -180 and 180"}), 400
    with state_lock:
        if lat_val is not None:
            state["gps_lat"] = lat_val
        if lon_val is not None:
            state["gps_lon"] = lon_val
        lat = state["gps_lat"]
        lon = state["gps_lon"]
    run_ai_detection()
    save_state_to_db()
    return jsonify({"success": True, "lat": lat, "lon": lon})

# ---------------------------------------------------------------------------
# FLASK API — Simulation Controls
# ---------------------------------------------------------------------------

@app.route("/api/simulate/accident", methods=["POST"])
def api_simulate_accident():
    """Trigger accident simulation for demonstration."""
    with state_lock:
        state["sensor_impact"]       = 9.2
        state["sensor_damage"]       = 8.5
        state["sensor_deceleration"] = 8.8
        state["sensor_fire"]         = 1
        state["sensor_smoke"]        = 2
    run_ai_detection()
    save_state_to_db()
    with state_lock:
        confidence = state["accident_confidence"]
        detected   = state["accident_detected"]
    log_event("SIMULATION", "Accident simulation triggered", confidence)
    return jsonify({
        "success":              True,
        "message":              "Accident simulation activated.",
        "accident_confidence":  confidence,
        "accident_detected":    detected,
    })

@app.route("/api/simulate/normal", methods=["POST"])
def api_simulate_normal():
    """Reset to normal operating state for demonstration."""
    with state_lock:
        state["sensor_impact"]       = 0.5
        state["sensor_damage"]       = 0.3
        state["sensor_deceleration"] = 0.2
        state["sensor_fire"]         = 0
        state["sensor_smoke"]        = 0
        state["accident_detected"]   = False
        state["accident_confidence"] = 0.0
        state["emergency_active"]    = False
        state["emergency_status"]    = "Standby"
        state["affected_zone"]       = "None"
        state["fire_status"]         = "NORMAL"
        state["smoke_status"]        = "NORMAL"
    save_state_to_db()
    log_event("SIMULATION", "Normal state simulation activated", 0.0)
    return jsonify({"success": True, "message": "Normal state simulation activated."})

# ---------------------------------------------------------------------------
# FLASK API — Voice (browser-based only, no phone calls)
# ---------------------------------------------------------------------------

@app.route("/api/voice/status", methods=["GET"])
def api_voice_status():
    """Returns current system status text for browser speech synthesis."""
    if state["accident_detected"]:
        msg = (
            f"Alert. Accident detected on bus {state['bus_id']}. "
            f"Accident confidence is {state['accident_confidence']} percent. "
            f"Total passengers onboard: {state['passengers_total']}. "
            f"Nearest hospital: {state['nearest_hospital']}, "
            f"approximately {state['hospital_distance_km']} kilometres away. "
            f"Emergency response is active."
        )
    else:
        msg = (
            f"Bus-Safe X status. Bus {state['bus_id']}. "
            f"System is operating normally. "
            f"Total passengers onboard: {state['passengers_total']}. "
            f"No accident detected."
        )
    return jsonify({"text": msg, "accident_detected": state["accident_detected"]})

# ---------------------------------------------------------------------------
# FLASK API — Event Log
# ---------------------------------------------------------------------------

@app.route("/api/events", methods=["GET"])
def api_events():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT timestamp, event_type, description, confidence FROM event_log ORDER BY id DESC LIMIT 20")
            rows = c.fetchall()
        events = [{"timestamp": r[0], "type": r[1], "description": r[2], "confidence": r[3]} for r in rows]
    except sqlite3.Error as e:
        print(f"[api_events] DB error: {e}", flush=True)
        events = []
    return jsonify(events)

# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    load_state_from_db()
    run_ai_detection()
    save_state_to_db()

    # Start background auto-save thread
    t = threading.Thread(target=auto_save_loop, daemon=True)
    t.start()

    print("")
    print("BUS-SAFE X -- Python Flask Server")
    print("=" * 60)
    print("Install : pip install flask")
    print("Run     : python app.py")
    print("Open    : http://localhost:5000")
    print("")
    print("State auto-saves to bussafe.db")
    print("=" * 60)
    print("")

    app.run(debug=False, host="0.0.0.0", port=5000)