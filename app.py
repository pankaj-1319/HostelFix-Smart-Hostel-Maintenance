from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import sqlite3
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DB = ROOT / "database" / "hostelfix.db"
MODELS = ROOT / "models"

app = Flask(__name__)
app.secret_key = "hostelfix-academic-demo"

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    DB.parent.mkdir(exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            room TEXT NOT NULL,
            block TEXT NOT NULL,
            complaint_type TEXT NOT NULL,
            location TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT,
            days_pending INTEGER DEFAULT 0,
            affected_rooms INTEGER DEFAULT 1,
            status TEXT DEFAULT 'Pending',
            created_at TEXT NOT NULL
        )
    """)
    if conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] == 0:
        sample = [
            ("Aarav","B-204","B","Water Leakage","Bathroom","High","Leakage near wash basin",2,3,"Pending"),
            ("Riya","C-112","C","Wi-Fi","Room","Medium","Weak network connection",1,5,"In Progress"),
            ("Kabir","A-308","A","Electricity","Room","Urgent","Power socket not working",3,2,"Pending"),
            ("Meera","D-105","D","Cleaning","Common Area","Low","Dustbin needs cleaning",0,1,"Resolved"),
            ("Dev","B-301","B","Fan/AC","Room","High","Ceiling fan making noise",2,1,"In Progress")
        ]
        conn.executemany("""
            INSERT INTO complaints
            (name,room,block,complaint_type,location,severity,description,
             days_pending,affected_rooms,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, [x + (datetime.now().strftime("%Y-%m-%d %H:%M"),) for x in sample])
    conn.commit()
    conn.close()

def load_models():
    try:
        return (
            joblib.load(MODELS/"j48_like.joblib"),
            joblib.load(MODELS/"naive_bayes.joblib"),
            joblib.load(MODELS/"linear_regression.joblib")
        )
    except Exception:
        return None, None, None

J48, NB, REG = load_models()

@app.context_processor
def globals_for_templates():
    return {"current_user": session.get("user")}

def rows():
    conn = get_db()
    data = conn.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()
    conn.close()
    return data

@app.route("/")
def dashboard():
    data = rows()
    types, severity = {}, {}
    for r in data:
        types[r["complaint_type"]] = types.get(r["complaint_type"], 0) + 1
        severity[r["severity"]] = severity.get(r["severity"], 0) + 1
    return render_template(
        "dashboard.html",
        total=len(data),
        pending=sum(r["status"]=="Pending" for r in data),
        progress=sum(r["status"]=="In Progress" for r in data),
        resolved=sum(r["status"]=="Resolved" for r in data),
        urgent=sum(r["severity"]=="Urgent" for r in data),
        types=types, priorities=severity, recent=data[:6]
    )

@app.route("/complaints")
def complaints():
    return render_template("complaints.html", complaints=rows())

@app.route("/complaints/new", methods=["GET","POST"])
def new_complaint():
    if request.method == "POST":
        values = (
            request.form["name"], request.form["room"], request.form["block"],
            request.form["complaint_type"], request.form["location"],
            request.form["severity"], request.form.get("description",""),
            int(request.form.get("days_pending",0)),
            int(request.form.get("affected_rooms",1)), "Pending",
            datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        conn = get_db()
        conn.execute("""
            INSERT INTO complaints
            (name,room,block,complaint_type,location,severity,description,
             days_pending,affected_rooms,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, values)
        conn.commit(); conn.close()
        flash("Complaint submitted successfully.", "success")
        return redirect(url_for("complaints"))
    return render_template("new_complaint.html")

@app.route("/complaints/<int:cid>/status/<status>", methods=["POST"])
def update_status(cid, status):
    if status not in ["Pending","In Progress","Resolved"]:
        return jsonify({"error":"Invalid status"}), 400
    conn = get_db()
    conn.execute("UPDATE complaints SET status=? WHERE id=?", (status,cid))
    conn.commit(); conn.close()
    return redirect(url_for("complaints"))

@app.route("/predict", methods=["GET","POST"])
def predict():
    result = None
    if request.method == "POST":
        payload = {
            "Hostel_Block": request.form["block"],
            "Complaint_Type": request.form["complaint_type"],
            "Location": request.form["location"],
            "Severity": request.form["severity"],
            "Season": request.form["season"],
            "Days_Pending": float(request.form["days_pending"]),
            "Previous_Complaints": float(request.form["previous_complaints"]),
            "Affected_Rooms": float(request.form["affected_rooms"]),
            "Staff_Available": float(request.form["staff_available"])
        }
        X = pd.DataFrame([payload])
        if J48 and NB and REG:
            priority = J48.predict(X)[0]
            alt = NB.predict(X)[0]
            resolution = max(0.5, float(REG.predict(X)[0]))
        else:
            priority = payload["Severity"]; alt = payload["Severity"]; resolution = 2.0
        risk = "High" if payload["Previous_Complaints"] >= 5 or payload["Days_Pending"] >= 3 else "Low"
        result = {
            "priority": priority, "alt": alt, "resolution": round(resolution,1),
            "risk": risk,
            "recommendation": "Assign maintenance staff immediately." if priority in ["High","Urgent"]
            else "Schedule the complaint in the next maintenance round."
        }
    return render_template("predict_pro.html", result=result)

@app.route("/analytics")
def analytics():
    data = rows()
    types, status = {}, {"Pending":0,"In Progress":0,"Resolved":0}
    for r in data:
        types[r["complaint_type"]] = types.get(r["complaint_type"],0)+1
        status[r["status"]] = status.get(r["status"],0)+1
    return render_template("analytics_pro.html", type_counts=types, status_counts=status)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["email"].split("@")[0].title()
        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/api/stats")
def api_stats():
    data = rows()
    return jsonify({
        "total": len(data),
        "pending": sum(r["status"]=="Pending" for r in data),
        "resolved": sum(r["status"]=="Resolved" for r in data),
        "urgent": sum(r["severity"]=="Urgent" for r in data)
    })

init_db()

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
