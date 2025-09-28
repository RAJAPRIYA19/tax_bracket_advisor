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
        criterion="entropy", max_depth=VC_DIM,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42
    )
    scores_old = cross_val_score(model_old, X, y_old, cv=5)
    ml_accuracy_old = scores_old.mean()
    model_old.fit(X, y_old)

    # New model
    y_new = df['new_bracket'].astype(str)
    model_new = DecisionTreeClassifier(
        criterion="entropy", max_depth=VC_DIM,
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

def make_tree_plot(model, feature_names, class_names, max_depth=3):
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
    global VC_DIM
    if "VC_DIM" in session:
        VC_DIM = session["VC_DIM"]
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
            tree_png=make_tree_plot(ml_model_old,['age','income','total_deductions'],class_names,max_depth=VC_DIM)
            feature_importances=dict(zip(['age','income','total_deductions'],ml_model_old.feature_importances_))
    elif suggested_regime == "new":
        with prolog_lock:
            q = list(prolog.query("taxable_income(new, TI), bracket_from_ti(new, TI, B)."))
        prolog_pred = str(q[0]['B']) if q else None
        if ml_model_new is not None:
            Xnew=pd.DataFrame([[age,income,total_deductions]],columns=['age','income','total_deductions'])
            ml_pred=ml_model_new.predict(Xnew)[0]; agreement=(ml_pred==prolog_pred); accuracy=ml_accuracy_new
            df=pd.read_csv(DATA_CSV); class_names=sorted(df['new_bracket'].astype(str).unique())
            tree_png=make_tree_plot(ml_model_new,['age','income','total_deductions'],class_names,max_depth=VC_DIM)
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
        def diagnose_str(train_acc, cv_mean, cv_std, vc, n_samples):
            # thresholds (stricter)
            underfit_thresh = 0.75      # below this we consider poor fit
            gap_overfit_thresh = 0.03   # >3% train-cv gap = overfit
            train_very_high = 0.99      # near perfect on train
            cv_std_high = 0.06          # unstable folds
            complexity_multiplier = 1.5

            # Underfitting check
            if train_acc < underfit_thresh and cv_mean < underfit_thresh:
                return "❌ Underfitting (model too simple)"

            # Overfitting checks
            gap = train_acc - cv_mean
            if gap > gap_overfit_thresh:
                return f"⚠️ Overfitting (train-cv gap = {gap:.2%})"

            if train_acc >= train_very_high and cv_mean < 0.98:
                return "⚠️ Overfitting"

            if cv_std > cv_std_high:
                return f"⚠️ High variance across folds (cv std = {cv_std:.3f}) — possible overfit/instability"

            # Complexity heuristic (relative to dataset size)
            if vc > complexity_multiplier * math.log2(max(2, n_samples)):
                return "⚠️ High VC relative to dataset size — risk of overfitting"

            # Otherwise accept as good
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
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
