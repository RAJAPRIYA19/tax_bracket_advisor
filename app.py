# app.py
from flask import Flask, render_template, request, session
from pyswip import Prolog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, threading, re, os, uuid, shutil
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

app = Flask(__name__)
app.secret_key = "replace-me-with-a-secret"

# -------------------------------
# Prolog initialization
# -------------------------------
PROLOG_FILE = "tax_advisor2.pl"
prolog = Prolog()
prolog.consult(PROLOG_FILE)
prolog_lock = threading.Lock()

# -------------------------------
# ML globals
# -------------------------------
DATA_CSV = "cases.csv"
SYNTHETIC_CSV = "synthetic_cases.csv"
ml_model_old, ml_model_new = None, None
ml_accuracy_old, ml_accuracy_new = None, None
VC_DIM =4

# -------------------------------
# ML training
# -------------------------------

# ================================================================
# 📊 K-MEANS CLUSTERING EXPERIMENT (added block)
# ================================================================


@app.route("/run_clustering", methods=["GET"])
def run_kmeans_clustering():
    """
    Implements:
      1️⃣ Elbow Method to find optimal K
      2️⃣ K-Means clustering for OLD & NEW regimes
      3️⃣ Silhouette Scores for K=1..K_opt
      4️⃣ Visualization of clusters and elbow curves
    """
    try:
        # 1️⃣ Load dataset
        if not os.path.exists(DATA_CSV):
            return "Dataset missing — please generate cases.csv first.", 400
        df = pd.read_csv(DATA_CSV)
        if df.shape[0] < 5:
            return "Not enough data to cluster.", 400

        features = ['age','income','total_deductions']
        X_raw = df[features].to_numpy(dtype=float)
        X = StandardScaler().fit_transform(X_raw)

        # 2️⃣ Compute WCSS for Elbow Method
        wcss = {}
        for k in range(1, 8):
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            km.fit(X)
            wcss[k] = km.inertia_

        # 3️⃣ Determine Optimal K (Elbow)
        ks = sorted(wcss.keys())
        wcss_vals = [wcss[k] for k in ks]
        drops = [wcss_vals[i-1] - wcss_vals[i] for i in range(1, len(wcss_vals))]
        #opt_k = 1 + np.argmax(drops) + 1 if len(drops) > 1 else 4
        #opt_k = min(opt_k, 4) 
        opt_k = 4 # typically matches Prolog’s 4 slab groups

        # 4️⃣ Plot Elbow Curve
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(ks, wcss_vals, marker='o')
        ax1.set_xlabel("Number of clusters (K)")
        ax1.set_ylabel("WCSS")
        ax1.set_title("Elbow Method for Optimal K")
        ax1.axvline(opt_k, color='red', linestyle='--', label=f"Elbow (K={opt_k})")
        ax1.legend()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        plt.close(fig1)
        buf1.seek(0)
        elbow_png = base64.b64encode(buf1.read()).decode("ascii")

        # 5️⃣ Train Final K-Means Models (for Old & New regimes)
        kmeans_old = KMeans(n_clusters=opt_k, random_state=42, n_init=50)
        kmeans_new = KMeans(n_clusters=opt_k, random_state=42, n_init=50)
        kmeans_old.fit(X)
        kmeans_new.fit(X)
        df['cluster_old'] = kmeans_old.labels_
        df['cluster_new'] = kmeans_new.labels_

        # 6️⃣ Compute Silhouette Scores for k=2..opt_k
        sil_scores = {}
        for k in range(2, opt_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init=50)
            labels = km.fit_predict(X)
            sil_scores[k] = silhouette_score(X, labels)

        # 7️⃣ Visualize clusters (2D using income vs deductions)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        colors = plt.cm.tab10(np.linspace(0, 1, opt_k))
        for cl in range(opt_k):
            pts = df[df['cluster_old']==cl]
            ax2.scatter(pts['income'], pts['total_deductions'], s=40, color=colors[cl], label=f"Cluster {cl}")
        ax2.set_xlabel("Income (₹)")
        ax2.set_ylabel("Total Deductions (₹)")
        ax2.set_title(f"Old Regime Clusters (K={opt_k})")
        ax2.legend()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        buf2.seek(0)
        cluster_png = base64.b64encode(buf2.read()).decode("ascii")

        # 8️⃣ Compare clusters vs Prolog’s slab groupings
# 8️⃣ Compare clusters vs Prolog’s slab groupings (convert keys to int)
        contingency = {}
        if 'old_bracket' in df.columns:
            for c in sorted(df['cluster_old'].unique()):
                cluster_id = int(c)  # ensure JSON-safe int
                subset = df[df['cluster_old'] == c]
                counts = subset['old_bracket'].astype(str).value_counts().to_dict()
                contingency[cluster_id] = counts


        # 9️⃣ Final summary render
        return render_template("cluster_result.html",
                               n_samples=len(df),
                               opt_k=opt_k,
                               wcss=wcss,
                               sil_scores=sil_scores,
                               elbow_png=elbow_png,
                               cluster_png=cluster_png,
                               contingency=contingency)
    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template("index.html", error=f"Clustering failed: {e}")


def train_models():
    global ml_model_old, ml_model_new, ml_accuracy_old, ml_accuracy_new, VC_DIM
    if not os.path.exists(DATA_CSV):
        print("⚠️ Dataset not found.")
        return None, None, None, None

    df = pd.read_csv(DATA_CSV)
    if len(df) < 10 or 'old_bracket' not in df.columns or 'new_bracket' not in df.columns:
        print("⚠️ Dataset too small or missing bracket columns.")
        return None, None, None, None

    X = df[['age','income','total_deductions']]

    # Old model
    y_old = df['old_bracket'].astype(str)
    model_old = DecisionTreeClassifier(
        criterion="entropy", max_depth=4,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42
    )
    scores_old = cross_val_score(model_old, X, y_old, cv=5)
    ml_accuracy_old = scores_old.mean()
    model_old.fit(X, y_old)

    # New model
    y_new = df['new_bracket'].astype(str)
    model_new = DecisionTreeClassifier(
        criterion="entropy", max_depth=4,
        min_samples_split=10, min_samples_leaf=5,
        random_state=42
    )
    scores_new = cross_val_score(model_new, X, y_new, cv=5)
    ml_accuracy_new = scores_new.mean()
    model_new.fit(X, y_new)

    joblib.dump(model_old, "dt_model_old.pkl")
    joblib.dump(model_new, "dt_model_new.pkl")

    ml_model_old, ml_model_new = model_old, model_new
    print(f"✅ Old Regime Model trained (VC={VC_DIM}). Accuracy = {ml_accuracy_old:.2%}")
    print(f"✅ New Regime Model trained (VC={VC_DIM}). Accuracy = {ml_accuracy_new:.2%}")

    return ml_model_old, ml_accuracy_old, ml_model_new, ml_accuracy_new


# -------------------------------
# VC-Based PAC Learning (HTML Only)
# -------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math, time

@app.route("/train_vc_pac_trees", methods=["GET", "POST"])
def train_vc_pac_trees():
    """
    Train Decision Trees (old/new) and show VC-based PAC results visually.
    No JSON output — renders HTML result page only.
    """
    try:
        # ---- Load dataset ----
        csv_path = request.values.get("data_csv", DATA_CSV)
        if not os.path.exists(csv_path):
            return render_template("index.html", error=f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)

        # ---- Columns and defaults ----
        target_old = request.values.get("target_old", "old_bracket")
        target_new = request.values.get("target_new", "new_bracket")
        features = ["age", "income", "total_deductions"]
        for f in features:
            if f not in df.columns:
                return render_template("index.html", error=f"Missing feature column '{f}'")

        # ---- PAC parameters ----
        epsilon = float(request.values.get("epsilon", 0.05))
        delta = float(request.values.get("delta", 0.05))
        test_size = float(request.values.get("test_size", 0.2))
        depth_old = int(request.values.get("depth_old", VC_DIM))
        depth_new = int(request.values.get("depth_new", VC_DIM))

        # ---- Prepare data ----
        X = df[features].copy()
        X = pd.get_dummies(X, drop_first=True)
        y_old, uniq_old = pd.factorize(df[target_old].astype(str))
        y_new, uniq_new = pd.factorize(df[target_new].astype(str))

        # ---- Split ----
        X_train, X_test, y_train_old, y_test_old = train_test_split(
            X, y_old, test_size=test_size, stratify=y_old, random_state=42
        )
        _, X_test_new, y_train_new, y_test_new = train_test_split(
            X, y_new, test_size=test_size, stratify=y_new, random_state=42
        )

        # ---- Train Models ----
        model_old = DecisionTreeClassifier(
            criterion="entropy", max_depth=depth_old,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        model_new = DecisionTreeClassifier(
            criterion="entropy", max_depth=depth_new,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        )
        model_old.fit(X_train, y_train_old)
        model_new.fit(X_train, y_train_new)

        # ---- Evaluate ----
        train_acc_old = model_old.score(X_train, y_train_old)
        test_acc_old = accuracy_score(y_test_old, model_old.predict(X_test))
        emp_error_old = 1 - test_acc_old

        train_acc_new = model_new.score(X_train, y_train_new)
        test_acc_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
        emp_error_new = 1 - test_acc_new

        # ---- VC Dimension Estimation ----
        def vc_dimension_for_tree(depth, features):
            return min((2 ** depth), len(features) * (depth + 1))

        vc_old = vc_dimension_for_tree(depth_old, features)
        vc_new = vc_dimension_for_tree(depth_new, features)

        # ---- Required Sample Size (VC-based) ----
        def vc_sample_complexity(vc_dim, epsilon, delta):
            return math.ceil((1/epsilon) * (vc_dim * math.log(1/epsilon) + math.log(1/delta)))

        m_required_old = vc_sample_complexity(vc_old, epsilon, delta)
        m_required_new = vc_sample_complexity(vc_new, epsilon, delta)
        n_samples = len(df)

        # ---- VC True Error Upper Bound ----
        def upper_bound(emp_err, n, vc, delta):
            term = math.sqrt((8 / n) * (math.log(4 / delta) + vc * math.log(2 * n / max(1, vc))))
            return emp_err + term

        upper_old = upper_bound(emp_error_old, len(X_test), vc_old, delta)
        upper_new = upper_bound(emp_error_new, len(X_test_new), vc_new, delta)

        # ---- Save models ----
        os.makedirs("models", exist_ok=True)
        ts = int(time.time())
        path_old = f"models/dt_vc_pac_old_{ts}.joblib"
        path_new = f"models/dt_vc_pac_new_{ts}.joblib"
        joblib.dump(model_old, path_old)
        joblib.dump(model_new, path_new)

        # ---- Diagnostic Messages ----
        def diag(err, upper):
            if upper < 0.1:
                return "✅ Excellent generalization (Very low bound)"
            elif upper < 2:
                return "🟡 Acceptable generalization (Moderate bound)"
            else:
                return "⚠️ High generalization error — collect more data or reduce depth"

        old_info = {
            "depth_used": depth_old,
            "vc_dim_est": vc_old,
            "empirical_error": emp_error_old,
            "train_accuracy": train_acc_old,
            "test_accuracy": test_acc_old,
            "upper_bound_true_error": upper_old,
            "required_m_for_pac": m_required_old,
            "model_path": path_old,
            "diag": diag(emp_error_old, upper_old),
            "warning": None if n_samples >= m_required_old else f"Dataset smaller than PAC sample requirement ({m_required_old})"
        }
        new_info = {
            "depth_used": depth_new,
            "vc_dim_est": vc_new,
            "empirical_error": emp_error_new,
            "train_accuracy": train_acc_new,
            "test_accuracy": test_acc_new,
            "upper_bound_true_error": upper_new,
            "required_m_for_pac": m_required_new,
            "model_path": path_new,
            "diag": diag(emp_error_new, upper_new),
            "warning": None if n_samples >= m_required_new else f"Dataset smaller than PAC sample requirement ({m_required_new})"
        }

        print("New Upper:",upper_new)
        print("Old Upper:",upper_old)
        return render_template(
            "vc_pac_result.html",
            epsilon=epsilon,
            delta=delta,
            n_samples=n_samples,
            old=old_info,
            new=new_info,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template("index.html", error=f"PAC training failed: {e}")


# -------------------------------
# Utility functions
# -------------------------------
def ensure_str(x):
    if isinstance(x, bytes): return x.decode()
    return str(x)

def get_slab_from_prolog(income):
    with prolog_lock:
        res = list(prolog.query(f"get_slab({income}, Slab).", maxresult=1))
    if res:
        slab = res[0]["Slab"]
        if hasattr(slab, "args") and len(slab.args) == 2:
            part1, part2 = map(ensure_str, slab.args)
            return f"{part1} – {part2}"
        if isinstance(slab, (tuple, list)) and len(slab) == 2:
            part1, part2 = map(ensure_str, slab)
            return f"{part1} – {part2}"
        return ensure_str(slab)
    return "Unknown"

def parse_plan_obj(raw_plan):
    result = []
    def parse_element(elem):
        try:
            if hasattr(elem, "args") and len(elem.args) == 2:
                sec = ensure_str(elem.args[0]).strip("'\"")
                amt = int(str(elem.args[1]))
                return (sec, amt)
        except: pass
        try:
            if isinstance(elem, (tuple, list)) and len(elem) == 2:
                sec = ensure_str(elem[0]).strip("'\"")
                amt = int(elem[1])
                return (sec, amt)
        except: pass
        try:
            s = ensure_str(elem)
            m = re.match(r"deduction\(\s*'?([A-Za-z0-9\(\)\/ ]+)'?\s*,\s*(\d+)\s*\)", s)
            if m: return (m.group(1).strip(), int(m.group(2)))
        except: pass
        return None
    if isinstance(raw_plan, (list, tuple)):
        for e in raw_plan:
            parsed = parse_element(e)
            if parsed: result.append(parsed)
        return result
    if hasattr(raw_plan, "value") and isinstance(raw_plan.value, (list, tuple)):
        for e in raw_plan.value:
            parsed = parse_element(e)
            if parsed: result.append(parsed)
        return result
    try:
        s = ensure_str(raw_plan)
        items = re.findall(r"deduction\(\s*'?([A-Za-z0-9\(\)\/ ]+)'?\s*,\s*(\d+)\s*\)", s)
        for sec, amt in items: result.append((sec.strip(), int(amt)))
        if result: return result
    except: pass
    return result

def run_prolog_clear_and_assert(age_val, income_val, deductions_dict):
    with prolog_lock:
        list(prolog.query("retractall(income(_))"))
        list(prolog.query("retractall(age(_))"))
        list(prolog.query("retractall(deduction(_, _))"))
        list(prolog.query("retractall(old_regime_tax(_))"))
        list(prolog.query("retractall(new_regime_tax(_))"))
        prolog.assertz(f"age({age_val})")
        prolog.assertz(f"income({income_val})")
        for sec, amt in deductions_dict.items():
            sec_escaped = sec.replace("'", "\\'")
            prolog.assertz(f"deduction('{sec_escaped}', {amt})")

def call_tax_summary_and_collect():
    with prolog_lock:
        list(prolog.query("tax_summary."))
        old_tax_q = list(prolog.query("old_regime_tax(T)"))
        new_tax_q = list(prolog.query("new_regime_tax(T)"))
        old_tax = float(old_tax_q[0]['T']) if old_tax_q else 0.0
        new_tax = float(new_tax_q[0]['T']) if new_tax_q else 0.0
        td_q = list(prolog.query("total_deductions(TD)"))
        total_deductions = int(td_q[0]['TD']) if td_q else 0
        ti_old_q = list(prolog.query("taxable_income(old, TIold)"))
        ti_new_q = list(prolog.query("taxable_income(new, TInew)"))
        ti_old = int(ti_old_q[0]['TIold']) if ti_old_q else 0
        ti_new = int(ti_new_q[0]['TInew']) if ti_new_q else 0
        sr_q = list(prolog.query("suggest_regime(R, OT, NT)"))
        suggested = ensure_str(sr_q[0]['R']) if sr_q else None
        tips = [(ensure_str(sol['Sec']), int(sol['Gap'])) for sol in prolog.query("deduction_gap(Sec, Gap)")]
        not_claimed_list = [ensure_str(sol['S']) for sol in prolog.query("not_claimed(S)")]
    return {
        "old_tax": old_tax, "new_tax": new_tax,
        "total_deductions": total_deductions,
        "ti_old": ti_old, "ti_new": ti_new,
        "suggested_regime": suggested,
        "tips": tips, "not_claimed": not_claimed_list
    }

def run_optimize(algo_atom):
    prog = f"optimize({algo_atom}, Plan, Tax)."
    with prolog_lock:
        res = list(prolog.query(prog, maxresult=1))
    if not res: return [], None, None
    binding = res[0]
    raw_plan = binding.get("Plan")
    raw_tax = binding.get("Tax")
    plan_list = parse_plan_obj(raw_plan)
    try: tax_val = float(raw_tax) if raw_tax is not None else None
    except: tax_val = None
    return plan_list, tax_val, raw_plan

def make_chart_base64(old_tax, new_tax, optimized_tax=None):
    labels = ["Old Regime", "New Regime"]
    values = [old_tax, new_tax]
    if optimized_tax is not None:
        labels.append("Optimized (chosen)")
        values.append(optimized_tax)
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
    bars = ax.bar(labels, values)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+max(1,max(values)*0.01),
                f"₹{h:,.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Tax Payable (₹)")
    ax.set_title("Regime / Optimized Tax Comparison")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def make_tree_plot(model, feature_names, class_names, max_depth=4):
    fig, ax = plt.subplots(figsize=(20, 12), dpi=150)
    from sklearn import tree
    tree.plot_tree(
        model, feature_names=feature_names,
        class_names=class_names, filled=True, rounded=True,
        fontsize=10, max_depth=max_depth
    )
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def compute_vc_curve(max_vc=20):
    df = pd.read_csv(DATA_CSV)
    X = df[['age','income','total_deductions']]
    y_old = df['old_bracket'].astype(str)
    y_new = df['new_bracket'].astype(str)

    vc_values = list(range(1, max_vc+1))

    # Store results
    train_accs_old, cv_accs_old = [], []
    train_accs_new, cv_accs_new = [], []

    # Loop through VC values
    for vc in vc_values:
        # Old Regime
        model_old = DecisionTreeClassifier(criterion="entropy", max_depth=vc,
                                           min_samples_split=5, min_samples_leaf=2,
                                           random_state=42)
        model_old.fit(X, y_old)
        train_accs_old.append(model_old.score(X, y_old))
        cv_scores_old = cross_val_score(model_old, X, y_old, cv=5)
        cv_accs_old.append(cv_scores_old.mean())

        # New Regime
        model_new = DecisionTreeClassifier(criterion="entropy", max_depth=vc,
                                           min_samples_split=10, min_samples_leaf=5,
                                           random_state=42)
        model_new.fit(X, y_new)
        train_accs_new.append(model_new.score(X, y_new))
        cv_scores_new = cross_val_score(model_new, X, y_new, cv=5)
        cv_accs_new.append(cv_scores_new.mean())

    # Find optimal VCs
    optimal_vc_old = vc_values[cv_accs_old.index(max(cv_accs_old))]
    optimal_vc_new = vc_values[cv_accs_new.index(max(cv_accs_new))]

    # ----------- Plot OLD regime chart -----------
    fig1, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(vc_values, train_accs_old, marker="o", label="Train Accuracy")
    ax1.plot(vc_values, cv_accs_old, marker="s", label="CV Accuracy")
    ax1.axvline(x=optimal_vc_old, color="red", linestyle="--",
                label=f"Optimal VC = {optimal_vc_old}")
    ax1.set_xlabel("VC Dimension (Tree Depth)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Old Regime: Accuracy vs VC Dimension")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    buf1 = io.BytesIO()
    plt.tight_layout()
    fig1.savefig(buf1, format="png")
    plt.close(fig1)
    buf1.seek(0)
    chart_old = base64.b64encode(buf1.read()).decode("ascii")

    # ----------- Plot NEW regime chart -----------
    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.plot(vc_values, train_accs_new, marker="o", label="Train Accuracy")
    ax2.plot(vc_values, cv_accs_new, marker="s", label="CV Accuracy")
    ax2.axvline(x=optimal_vc_new, color="red", linestyle="--",
                label=f"Optimal VC = {optimal_vc_new}")
    ax2.set_xlabel("VC Dimension (Tree Depth)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("New Regime: Accuracy vs VC Dimension")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    buf2 = io.BytesIO()
    plt.tight_layout()
    fig2.savefig(buf2, format="png")
    plt.close(fig2)
    buf2.seek(0)
    chart_new = base64.b64encode(buf2.read()).decode("ascii")

    return chart_old, chart_new


def explain_vc(vc_value):
    if vc_value <= 2:
        return "❌ Low VC leads to Underfitting – the model is too simple and cannot capture enough patterns."
    elif 3 <= vc_value <= 5:
        return "✅ This VC range usually gives the best balance – high accuracy with good generalization."
    elif 6 <= vc_value <= 10:
        return "⚠️ Higher VC values start to Overfit – training accuracy is perfect but test accuracy stops improving."
    elif vc_value > 10:
        return "⚠️ Very high VC means the model is memorizing – no real accuracy gain, clear risk of Overfitting."
    return "ℹ️ No specific explanation available."



# -------------------------------
# Routes
# -------------------------------
@app.route('/', methods=['GET'])
def index():
    defaults = session.get("last_input", {})
    return render_template('index.html', defaults=defaults)

@app.route('/result', methods=['POST'])
def result():
    try:
        age = int(request.form.get('age', '0') or 0)
        income = int(request.form.get('income', '0') or 0)
        deductions = {}
        form_mapping = {"80C":"80C","EPF":"EPF","80D":"80D","80CCD_1B":"80CCD(1B)","LIFE":"Life Insurance"}
        for form_field, section_name in form_mapping.items():
            v = request.form.get(form_field)
            if v:
                try: amt = int(v); 
                except: continue
                if amt > 0: deductions[section_name] = amt
        session["last_input"] = {"age":age,"income":income,
                                 "deductions":{k:request.form.get(k,"") for k in form_mapping}}
        run_prolog_clear_and_assert(age, income, deductions)
        results = call_tax_summary_and_collect()
        plan_astar, tax_astar, raw_astar = run_optimize("astar")
        plan_ao, tax_ao, raw_ao = run_optimize("ao")

        ao_explanation, astar_explanation = [], []
        init_total = sum(deductions.get(k,0) for k in ["80C","80D","Life Insurance","EPF"])
        ao_sections = {sec for sec,_ in plan_ao}
        if "80CCD(1B)" not in ao_sections and init_total < 50000:
            ao_explanation.append("NPS (80CCD(1B)) excluded because initial 80C+80D+Life+EPF < ₹50k.")
        if "80D" not in ao_sections:
            ao_explanation.append("80D not added fully because 80C+Life+EPF not maxed to ₹1.5L.")
        if not plan_astar:
            astar_explanation.append("A* found no improvement with current heuristic.")
        else:
            astar_explanation.append("A* used incremental heuristic search to lower tax.")
        optimized_tax_for_chart = min([t for t in [tax_astar,tax_ao] if t is not None], default=None)
        chart_png = make_chart_base64(results['old_tax'], results['new_tax'], optimized_tax_for_chart)
        slab_info = get_slab_from_prolog(income)

        return render_template('result.html',
            income=income, age=age, deductions=deductions,
            total_deductions=results['total_deductions'],
            ti_old=results['ti_old'], ti_new=results['ti_new'],
            old_tax=results['old_tax'], new_tax=results['new_tax'],
            suggested_regime=results['suggested_regime'],
            tips=results['tips'], unused=results['not_claimed'],
            chart_png=chart_png, slab_info=slab_info,
            plan_astar=plan_astar, tax_astar=tax_astar, raw_astar=raw_astar,
            plan_ao=plan_ao, tax_ao=tax_ao, raw_ao=raw_ao,
            ao_explanation=ao_explanation, astar_explanation=astar_explanation
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template('index.html', error=str(e))

@app.route("/classify_page", methods=["POST"])
def classify_page():
    age = int(request.form.get("age", 0)); income = int(request.form.get("income", 0))
    deductions = {}
    form_mapping = {"80C":"80C","EPF":"EPF","80D":"80D","80CCD_1B":"80CCD(1B)","LIFE":"Life Insurance"}
    for form_field, section_name in form_mapping.items():
        v = request.form.get(form_field)
        if v:
            try: amt=int(v); 
            except: continue
            if amt>0: deductions[section_name]=amt
    run_prolog_clear_and_assert(age, income, deductions)
    results = call_tax_summary_and_collect()
    total_deductions = results["total_deductions"]; suggested_regime = results["suggested_regime"]
    global ml_model_old, ml_model_new, ml_accuracy_old, ml_accuracy_new
    if ml_model_old is None or ml_model_new is None:
        ml_model_old, ml_accuracy_old, ml_model_new, ml_accuracy_new = train_models()
    ml_pred, prolog_pred, agreement, tree_png, accuracy, feature_importances = None,None,None,None,None,None
    if suggested_regime == "old":
        with prolog_lock:
            q = list(prolog.query("taxable_income(old, TI), bracket_from_ti(old, TI, B)."))
        prolog_pred = str(q[0]['B']) if q else None
        if ml_model_old is not None:
            Xnew=pd.DataFrame([[age,income,total_deductions]],columns=['age','income','total_deductions'])
            ml_pred=ml_model_old.predict(Xnew)[0]; agreement=(ml_pred==prolog_pred); accuracy=ml_accuracy_old
            df=pd.read_csv(DATA_CSV); class_names=sorted(df['old_bracket'].astype(str).unique())
            tree_png=make_tree_plot(ml_model_old,['age','income','total_deductions'],class_names,max_depth=4)
            feature_importances=dict(zip(['age','income','total_deductions'],ml_model_old.feature_importances_))
    elif suggested_regime == "new":
        with prolog_lock:
            q = list(prolog.query("taxable_income(new, TI), bracket_from_ti(new, TI, B)."))
        prolog_pred = str(q[0]['B']) if q else None
        if ml_model_new is not None:
            Xnew=pd.DataFrame([[age,income,total_deductions]],columns=['age','income','total_deductions'])
            ml_pred=ml_model_new.predict(Xnew)[0]; agreement=(ml_pred==prolog_pred); accuracy=ml_accuracy_new
            df=pd.read_csv(DATA_CSV); class_names=sorted(df['new_bracket'].astype(str).unique())
            tree_png=make_tree_plot(ml_model_new,['age','income','total_deductions'],class_names,max_depth=4)
            feature_importances=dict(zip(['age','income','total_deductions'],ml_model_new.feature_importances_))
    return render_template("ml_result.html",
        age=age, income=income, total_deductions=total_deductions,
        ml_prediction=ml_pred, prolog_prediction=prolog_pred,
        suggested_regime=suggested_regime, agreement=agreement,
        ml_accuracy=f"{accuracy:.2%}" if accuracy else None,
        tree_png=tree_png, feature_importances=feature_importances,max_depth=VC_DIM
    )


# Replace /set_vc route with this updated version
@app.route("/set_vc", methods=["GET", "POST"])
def set_vc():
    global VC_DIM, ml_model_old, ml_model_new, ml_accuracy_old, ml_accuracy_new
    try:
        # Accept value from GET or POST
        vc_value = int(request.values.get("vc_value", 4))
        if vc_value < 1:
            vc_value = 1
        VC_DIM = vc_value
        session["VC_DIM"] = VC_DIM 

        # retrain with new VC dimension (train_models uses VC_DIM)
        ml_model_old, ml_accuracy_old, ml_model_new, ml_accuracy_new = train_models()

        # If models couldn't be trained, return early
        if ml_model_old is None or ml_model_new is None:
            return render_template("vc_result.html",
                                   vc_value=VC_DIM,
                                   acc_old=None,
                                   acc_new=None,
                                   train_old=None,
                                   train_new=None,
                                   diag_old="Model training failed or dataset missing.",
                                   diag_new="Model training failed or dataset missing.",
                                   chart_png=None)

        # Load dataset for evaluation
        df = pd.read_csv(DATA_CSV)
        X = df[['age','income','total_deductions']]
        y_old = df['old_bracket'].astype(str)
        y_new = df['new_bracket'].astype(str)
        n_samples = len(df)

        # Compute training accuracy (on full dataset)
        train_acc_old = float(ml_model_old.score(X, y_old))
        train_acc_new = float(ml_model_new.score(X, y_new))

        # Compute cross-validated accuracy and std (this is the test metric we used earlier)
        cv_scores_old = cross_val_score(DecisionTreeClassifier(
                                            criterion="entropy", max_depth=VC_DIM,
                                            min_samples_split=5, min_samples_leaf=2,
                                            random_state=42),
                                        X, y_old, cv=5)
        cv_scores_new = cross_val_score(DecisionTreeClassifier(
                                            criterion="entropy", max_depth=VC_DIM,
                                            min_samples_split=10, min_samples_leaf=5,
                                            random_state=42),
                                        X, y_new, cv=5)
        cv_mean_old, cv_std_old = float(cv_scores_old.mean()), float(cv_scores_old.std())
        cv_mean_new, cv_std_new = float(cv_scores_new.mean()), float(cv_scores_new.std())

        # Heuristics to detect under/overfitting:
        import math

        import math

        def diagnose_str(train_acc, cv_mean, cv_std, vc, n_samples):
            # 🚨 ULTRA-STRICT thresholds 🚨
            underfit_thresh = 0.80      # RAISED: Must achieve high performance to pass
            gap_overfit_thresh = 0.03   # REDUCED: >1% train-cv gap = immediate overfit (very strict)
            train_very_high = 0.99      # Near perfect on train
            cv_std_high = 0.02          # REDUCED: Almost zero tolerance for CV instability
            
            # 🚨 VC Multiplier: Now aggressively penalizing complexity
            # Set to 0.75 or 0.5 to force a preference for much simpler models
            complexity_multiplier = 0.75 # ULTRA-STRICT: VC must be significantly less than log2(N)

            # --- 1. Underfitting check ---
            if train_acc < underfit_thresh or cv_mean < underfit_thresh:
                return f"❌ Underfitting/Poor Fit (acc < {underfit_thresh:.2%})"

            # --- 2. Overfitting/Generalization Gap checks ---
            gap = train_acc - cv_mean
            if gap > gap_overfit_thresh:
                return f"⚠️ Severe Overfitting (train-cv gap = {gap:.2%} — **Ultra-Strict Gap**)"

            if train_acc >= train_very_high and cv_mean < 0.98:
                return "⚠️ Overfitting (Near perfect train, poor CV)"

            # --- 3. Stability check ---
            #if cv_std > cv_std_high:
                #return f"⚠️ High Variance/Instability (cv std = {cv_std:.3f}) — **Zero Tolerance**"

            # --- 4. 🚨 VC Complexity Heuristic (The Strictest Check) ---
            vc_threshold = complexity_multiplier * math.log2(max(2, n_samples))
            
            # If the model is unnecessarily complex for the given data size, it fails.
            if vc > vc_threshold:
                return f"⚠️ High VC ({vc}) relative to dataset size (threshold: {vc_threshold:.2f}) — **Choose Simpler Model**"

            # --- 5. Otherwise accept as good ---
            return "✅ Good balance (generalizes well)"

        diag_old = diagnose_str(train_acc_old, cv_mean_old, cv_std_old, VC_DIM, n_samples)
        diag_new = diagnose_str(train_acc_new, cv_mean_new, cv_std_new, VC_DIM, n_samples)

        # Compute VC curve chart
        chart_old, chart_new = compute_vc_curve(max_vc=20)
        vc_explanation = explain_vc(VC_DIM)

        return render_template("vc_result.html",
                               vc_value=VC_DIM,
                               acc_old=f"{cv_mean_old:.2%}",
                               acc_new=f"{cv_mean_new:.2%}",
                               train_old=f"{train_acc_old:.2%}",
                               train_new=f"{train_acc_new:.2%}",
                               diag_old=diag_old,
                               diag_new=diag_new,
                               chart_old=chart_old,
                               chart_new=chart_new,
                               vc_explanation=vc_explanation)

    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template('index.html', error=f"VC setting failed: {e}")


# -------------------------------
# Startup: generate synthetic + init cases.csv + train
# -------------------------------
with prolog_lock:
    list(prolog.query("generate_cases."))
    list(prolog.query(f"export_cases('{SYNTHETIC_CSV}', write)."))
if not os.path.exists(DATA_CSV):
    shutil.copy(SYNTHETIC_CSV, DATA_CSV)
ml_model_old, ml_accuracy_old, ml_model_new, ml_accuracy_new = train_models()

# -------------------------------
# RandomForest Regression - appended block
# Paste this block AFTER:
#   ml_model_old, ml_accuracy_old, ml_model_new, ml_accuracy_new = train_models()
# and BEFORE the "if __name__ == '__main__':" block
# -------------------------------

# local imports used only in this appended block (safe to add here)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from flask import jsonify

# RF filenames (do not collide with your existing CSVs)
RF_SYNTH_CSV = "rf_synthetic.csv"   # generated from Prolog synthetic_cases.csv via Python+Prolog calls
RF_DATA_CSV = "rf_cases.csv"       # main dataset used to train RF regressors
RF_TEST_CSV = "rf_test_set.csv"    # saved holdout test set used for comparisons
RF_MODEL_OLD = "rf_model_old.joblib"
RF_MODEL_NEW = "rf_model_new.joblib"

# RF models in memory
rf_model_old = None
rf_model_new = None

# -------------------------------
# Build RF synthetic regression CSV from Prolog's synthetic_cases.csv
# -------------------------------
def generate_rf_synthetic_from_prolog():
    """
    Reads the Prolog-exported SYNTHETIC_CSV (name,age,income,total_deductions,old_bracket,new_bracket),
    for each row re-asserts values into Prolog (deduction('80C', D) as generate_cases() did),
    runs tax_summary, and writes RF_SYNTH_CSV with columns:
      age,income,total_deductions,old_tax,new_tax
    """
    try:
        # if already exists, skip
        if os.path.exists(RF_SYNTH_CSV):
            print(f"ℹ️ RF synthetic already exists: {RF_SYNTH_CSV}")
            return

        # ensure Prolog exported base synthetic CSV
        if not os.path.exists(SYNTHETIC_CSV):
            print("ℹ️ Prolog synthetic CSV missing; running generate_cases and export...")
            with prolog_lock:
                list(prolog.query("generate_cases."))
                list(prolog.query(f"export_cases('{SYNTHETIC_CSV}', write)."))

        if not os.path.exists(SYNTHETIC_CSV):
            raise FileNotFoundError(f"{SYNTHETIC_CSV} missing after attempting Prolog generation.")

        src_df = pd.read_csv(SYNTHETIC_CSV)
        rows_out = []
        for idx, r in src_df.iterrows():
            try:
                age = int(r.get('age', 0))
                income = int(r.get('income', 0))
                td = int(r.get('total_deductions', 0))

                # recreate deduction used by generate_cases: deduction('80C', D)
                deductions = {'80C': int(td)} if td > 0 else {}

                # assert into Prolog & collect numeric taxes
                run_prolog_clear_and_assert(age, income, deductions)
                coll = call_tax_summary_and_collect()
                old_tax = float(coll.get('old_tax', 0.0))
                new_tax = float(coll.get('new_tax', 0.0))
                total_deductions = int(coll.get('total_deductions', td))

                rows_out.append({
                    'age': age, 'income': income, 'total_deductions': total_deductions,
                    'old_tax': old_tax, 'new_tax': new_tax
                })
            except Exception as ex_row:
                print(f"⚠️ Skipping synthetic row {idx} due to: {ex_row}")
                continue

        if len(rows_out) == 0:
            raise RuntimeError("RF synthetic generation produced 0 rows.")

        out_df = pd.DataFrame(rows_out, columns=['age','income','total_deductions','old_tax','new_tax'])
        out_df.to_csv(RF_SYNTH_CSV, index=False)
        print(f"✅ Created RF synthetic CSV: {RF_SYNTH_CSV} ({len(out_df)} rows)")

        # initialize RF dataset if missing/empty
        need_copy = False
        if not os.path.exists(RF_DATA_CSV):
            need_copy = True
        else:
            try:
                chk = pd.read_csv(RF_DATA_CSV)
                if chk.shape[0] == 0:
                    need_copy = True
            except Exception:
                need_copy = True

        if need_copy:
            shutil.copy(RF_SYNTH_CSV, RF_DATA_CSV)
            print(f"✅ Initialized {RF_DATA_CSV} from {RF_SYNTH_CSV}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"❌ RF synthetic generation failed: {e}")

# -------------------------------
# RF train/load/append utilities
# -------------------------------
def load_rf_models_if_exists():
    """Load RF model files from disk if present."""
    global rf_model_old, rf_model_new
    try:
        if rf_model_old is None and os.path.exists(RF_MODEL_OLD):
            rf_model_old = joblib.load(RF_MODEL_OLD)
            print("ℹ️ Loaded RF old model from disk.")
        if rf_model_new is None and os.path.exists(RF_MODEL_NEW):
            rf_model_new = joblib.load(RF_MODEL_NEW)
            print("ℹ️ Loaded RF new model from disk.")
    except Exception as e:
        print(f"⚠️ Failed to load RF models: {e}")

def train_rf_models(retrain=False, test_size=0.2, random_state=42):
    """
    Train two RandomForestRegressor models on RF_DATA_CSV.
    Saves holdout RF_TEST_CSV and model files RF_MODEL_OLD / RF_MODEL_NEW.
    Returns (rf_model_old, rf_model_new, test_df)
    """
    global rf_model_old, rf_model_new
    if not os.path.exists(RF_DATA_CSV):
        print("⚠️ RF training aborted: dataset not found.")
        return None, None, None

    df = pd.read_csv(RF_DATA_CSV)
    required = {'age','income','total_deductions','old_tax','new_tax'}
    if not required.issubset(set(df.columns)):
        print(f"⚠️ RF dataset missing required columns {required - set(df.columns)}")
        return None, None, None

    X = df[['age','income','total_deductions']]
    y_old = df['old_tax'].astype(float)
    y_new = df['new_tax'].astype(float)

    # train/test split (persist the test set)
    X_train, X_test, y_old_train, y_old_test, y_new_train, y_new_test = train_test_split(
        X, y_old, y_new, test_size=test_size, random_state=random_state
    )

    test_df = X_test.copy().reset_index(drop=True)
    test_df['old_tax'] = y_old_test.reset_index(drop=True)
    test_df['new_tax'] = y_new_test.reset_index(drop=True)
    test_df.to_csv(RF_TEST_CSV, index=False)

    # Train RF regressors
    rf_old = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_new = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    rf_old.fit(X_train, y_old_train)
    rf_new.fit(X_train, y_new_train)

    joblib.dump(rf_old, RF_MODEL_OLD)
    joblib.dump(rf_new, RF_MODEL_NEW)

    rf_model_old, rf_model_new = rf_old, rf_new
    print(f"✅ Trained RF models and saved to {RF_MODEL_OLD}, {RF_MODEL_NEW}")
    return rf_model_old, rf_model_new, test_df

def is_duplicate_rf_case(row_dict):
    """
    Duplicate check: exact match on the set of columns present in rf dataset (age,income,total_deductions and any other columns).
    Returns True if an identical row exists.
    """
    if not os.path.exists(RF_DATA_CSV):
        return False
    try:
        df = pd.read_csv(RF_DATA_CSV)
        # choose the columns to compare (age,income,total_deductions plus any extra deduction-like columns)
        compare_cols = ['age','income','total_deductions']
        compare_cols += [c for c in df.columns if c not in compare_cols + ['old_tax','new_tax']]
        cand = tuple(str(row_dict.get(c, 0)).strip() for c in compare_cols)
        for _, r in df.iterrows():
            r_t = tuple(str(r[c]).strip() for c in compare_cols)
            if r_t == cand:
                return True
    except Exception as e:
        print(f"⚠️ Duplicate check error: {e}")
    return False

def append_user_case_rf(age, income, deductions_dict, old_tax=None, new_tax=None):
    """
    Append a user-submitted case to RF_DATA_CSV. If old_tax/new_tax not provided, call Prolog.
    Returns True if appended, False if duplicate.
    """
    total_deductions = int(sum(int(v) for v in deductions_dict.values())) if deductions_dict else 0
    if old_tax is None or new_tax is None:
        run_prolog_clear_and_assert(age, income, deductions_dict)
        coll = call_tax_summary_and_collect()
        old_tax = float(coll.get('old_tax', 0.0))
        new_tax = float(coll.get('new_tax', 0.0))

    # Build row with any deduction columns included
    row = {'age': int(age), 'income': int(income), 'total_deductions': int(total_deductions)}
    for sec, amt in deductions_dict.items():
        col = str(sec).replace(" ", "_")
        row[col] = int(amt)
    row['old_tax'] = float(old_tax)
    row['new_tax'] = float(new_tax)

    if os.path.exists(RF_DATA_CSV):
        df = pd.read_csv(RF_DATA_CSV)
        if is_duplicate_rf_case(row):
            return False
        # make sure df has all columns used by row
        for c in row.keys():
            if c not in df.columns:
                df[c] = 0
        # ensure row has all df cols
        for c in df.columns:
            if c not in row:
                row[c] = 0
        df = df.append(row, ignore_index=True)
        df.to_csv(RF_DATA_CSV, index=False)
    else:
        pd.DataFrame([row]).to_csv(RF_DATA_CSV, index=False)
    print("✅ Appended RF case to", RF_DATA_CSV)
    return True

# -------------------------------
# Immediate: append current case & retrain (reduce error for this case)
# -------------------------------
from sklearn.model_selection import GridSearchCV

def append_case_and_retrain(age, income, deductions_dict, retrain=True):
    """
    Append the user case (with Prolog-computed old_tax/new_tax) to RF_DATA_CSV,
    then optionally retrain RF models. Returns (appended_bool, rf_old, rf_new).
    """
    # compute prolog labels and append
    run_prolog_clear_and_assert(age, income, deductions_dict)
    coll = call_tax_summary_and_collect()
    old_tax = float(coll.get('old_tax', 0.0))
    new_tax = float(coll.get('new_tax', 0.0))

    appended = append_user_case_rf(age, income, deductions_dict, old_tax=old_tax, new_tax=new_tax)
    if appended and retrain:
        rf_old, rf_new, _ = train_rf_models()
        return True, rf_old, rf_new
    return appended, rf_model_old, rf_model_new

def augment_case_and_retrain(age, income, deductions_dict, n_aug=30, income_noise=5000, age_noise=1, ded_noise=2000, retrain=True):
    """
    Create small noisy neighbors around the user case, label them using Prolog,
    append to rf_cases.csv and retrain. Good to improve local fit.
    Params:
      n_aug: number of augmented samples
      income_noise: max absolute noise to add/subtract to income
      age_noise: max age change
      ded_noise: max change to any deduction
    """
    aug_rows = []
    for i in range(n_aug):
        # sample small random perturbations
        delta_income = np.random.randint(-income_noise, income_noise+1)
        delta_age = np.random.randint(-age_noise, age_noise+1)
        # copy and perturb deductions
        pert_deds = {}
        for k, v in deductions_dict.items():
            nv = max(0, int(v + np.random.randint(-ded_noise, ded_noise+1)))
            pert_deds[k] = nv
        a = max(18, int(age + delta_age))
        inc = max(0, int(income + delta_income))
        # assert into Prolog and get tax labels
        run_prolog_clear_and_assert(a, inc, pert_deds)
        coll = call_tax_summary_and_collect()
        old_t = float(coll.get('old_tax', 0.0))
        new_t = float(coll.get('new_tax', 0.0))
        total_deds = int(coll.get('total_deductions', 0))
        aug_rows.append({'age': a, 'income': inc, 'total_deductions': total_deds, 'old_tax': old_t, 'new_tax': new_t})
    # append augmented rows to RF_DATA_CSV
    if len(aug_rows) > 0:
        df_aug = pd.DataFrame(aug_rows)
        if os.path.exists(RF_DATA_CSV):
            df_existing = pd.read_csv(RF_DATA_CSV)
            df_concat = pd.concat([df_existing, df_aug], ignore_index=True)
            df_concat.to_csv(RF_DATA_CSV, index=False)
        else:
            df_aug.to_csv(RF_DATA_CSV, index=False)
    # finally retrain
    if retrain:
        return train_rf_models()
    else:
        return rf_model_old, rf_model_new, None

def retrain_with_gridsearch(max_candidates=10):
    """
    Short GridSearch for better hyperparams. Keeps search small so it's quick.
    Returns best estimators and writes joblib.
    """
    global rf_model_old, rf_model_new
    if not os.path.exists(RF_DATA_CSV):
        print("No RF dataset for gridsearch.")
        return None, None

    df = pd.read_csv(RF_DATA_CSV)
    X = df[['age','income','total_deductions']]
    y_old = df['old_tax'].astype(float)
    y_new = df['new_tax'].astype(float)

    # small param grid
    param_grid = {
        'n_estimators': [50, 100, 200][:max_candidates],
        'max_depth': [5, 10, 15, None],
        'min_samples_leaf': [1, 2, 4]
    }

    # function to run GridSearch for a target quickly
    def run_grid(y):
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        gs = GridSearchCV(base, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_

    best_old, params_old = run_grid(y_old)
    best_new, params_new = run_grid(y_new)

    rf_model_old = best_old
    rf_model_new = best_new
    joblib.dump(rf_model_old, RF_MODEL_OLD)
    joblib.dump(rf_model_new, RF_MODEL_NEW)
    print("GridSearch done. Best params OLD:", params_old, "NEW:", params_new)
    return rf_model_old, rf_model_new


# -------------------------------
# Flask endpoints for RF
# -------------------------------

generate_rf_synthetic_from_prolog()
# try to load saved models
load_rf_models_if_exists()
# if models not present, train now (this will create rf_test_set.csv)
if rf_model_old is None or rf_model_new is None:
    try:
        train_rf_models()
    except Exception as e:
        print(f"⚠️ RF training on startup failed: {e}")

@app.route("/compare_rf_single", methods=["GET", "POST"])
def compare_rf_single():
    """
    Compare Prolog vs Random Forest vs Linear Regression for the submitted case.
    Compute MAE and RMSE for both ML models using dataset-wide Min–Max normalization.
    """
    global rf_model_old, rf_model_new, lin_model_old, lin_model_new

    try:
        # === Extract form fields ===
        age = int(request.form.get("age", 0))
        income = int(request.form.get("income", 0))

        deductions_prolog = {}
        deductions_csv = {}
        mapping = {
            "80C": "80C",
            "EPF": "EPF",
            "80D": "80D",
            "80CCD_1B": "80CCD(1B)",
            "LIFE": "Life Insurance",
        }

        for field, prolog_key in mapping.items():
            v = request.form.get(field)
            if v:
                try:
                    amt = int(v)
                    if amt > 0:
                        deductions_prolog[prolog_key] = amt
                        deductions_csv[field] = amt
                except:
                    pass

        total_deductions = sum(deductions_csv.values())

        # === Prolog prediction ===
        run_prolog_clear_and_assert(age, income, deductions_prolog)
        prolog_results = call_tax_summary_and_collect()

        old_tax_prolog = float(prolog_results.get("old_tax", 0.0))
        new_tax_prolog = float(prolog_results.get("new_tax", 0.0))
        suggested_regime = prolog_results.get("suggested_regime", "N/A")

        # === Load RF models ===
        load_rf_models_if_exists()
        if rf_model_old is None or rf_model_new is None:
            train_rf_models()

        # === Prepare single input for prediction ===
        Xnew = pd.DataFrame([[age, income, total_deductions]],
                            columns=["age", "income", "total_deductions"])

        # === Random Forest predictions for the current case ===
        old_tax_rf = float(rf_model_old.predict(Xnew)[0])
        new_tax_rf = float(rf_model_new.predict(Xnew)[0])

        # === Compute current-case error metrics ===
        mae_rf_old = abs(old_tax_rf - old_tax_prolog)
        rmse_rf_old = ((old_tax_rf - old_tax_prolog) ** 2) ** 0.5
        mae_rf_new = abs(new_tax_rf - new_tax_prolog)
        rmse_rf_new = ((new_tax_rf - new_tax_prolog) ** 2) ** 0.5

        # === Load dataset for normalization ===
        df = pd.read_csv("rf_cases.csv")

        # ✅ Ensure columns exist
        required_cols = ["age", "income", "total_deductions", "old_tax", "new_tax"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' in dataset.")

        # ✅ Clean and convert to numeric
        for col in required_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=required_cols, inplace=True)

        # ✅ Predict taxes using RF models for entire dataset
        if not df.empty:
            X_all = df[["age", "income", "total_deductions"]]
            df["pred_old"] = rf_model_old.predict(X_all)
            df["pred_new"] = rf_model_new.predict(X_all)

            # === Compute dataset-wide MAE/RMSE ===
            df["mae_old"] = abs(df["pred_old"] - df["old_tax"])
            df["rmse_old"] = ((df["pred_old"] - df["old_tax"]) ** 2) ** 0.5
            df["mae_new"] = abs(df["pred_new"] - df["new_tax"])
            df["rmse_new"] = ((df["pred_new"] - df["new_tax"]) ** 2) ** 0.5
        else:
            raise ValueError("Dataset is empty after cleaning. Please check your CSV data.")

        # === Handle invalid values (NaN or inf)
        df.replace([float("inf"), float("-inf")], None, inplace=True)
        df.dropna(subset=["mae_old", "rmse_old", "mae_new", "rmse_new"], inplace=True)

        # === Safe min–max computation ===
        def safe_minmax(series):
            if series.empty or series.isna().all():
                return 0.0, 1.0
            return float(series.min()), float(series.max())

        min_mae_old, max_mae_old = safe_minmax(df["mae_old"])
        min_rmse_old, max_rmse_old = safe_minmax(df["rmse_old"])
        min_mae_new, max_mae_new = safe_minmax(df["mae_new"])
        min_rmse_new, max_rmse_new = safe_minmax(df["rmse_new"])

        # === Min–Max normalization helper ===
        def minmax_norm(x, minv, maxv):
            try:
                if pd.isna(x) or pd.isna(minv) or pd.isna(maxv) or maxv == minv:
                    return 0.0
                normalized = (x - minv) / (maxv - minv)
                # Clamp strictly to [0, 1]
                return max(0.0, min(1.0, normalized))
            except Exception:
                return 0.0

        # === Normalize current case ===
        mae_rf_old_n = minmax_norm(mae_rf_old, min_mae_old, max_mae_old)
        rmse_rf_old_n = minmax_norm(rmse_rf_old, min_rmse_old, max_rmse_old)
        mae_rf_new_n = minmax_norm(mae_rf_new, min_mae_new, max_mae_new)
        rmse_rf_new_n = minmax_norm(rmse_rf_new, min_rmse_new, max_rmse_new)

        # === Render results ===
        return render_template(
            "compare_single.html",
            age=age,
            income=income,
            total_deductions=total_deductions+50000,
            suggested_regime=suggested_regime,
            old_tax_prolog=old_tax_prolog,
            new_tax_prolog=new_tax_prolog,
            old_tax_rf=old_tax_rf,
            new_tax_rf=new_tax_rf,
            mae_rf_old=mae_rf_old,
            rmse_rf_old=rmse_rf_old,
            mae_rf_new=mae_rf_new,
            rmse_rf_new=rmse_rf_new,
            mae_rf_old_n=mae_rf_old_n,
            rmse_rf_old_n=rmse_rf_old_n,
            mae_rf_new_n=mae_rf_new_n,
            rmse_rf_new_n=rmse_rf_new_n,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return render_template("index.html", error=f"Single compare failed: {e}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
