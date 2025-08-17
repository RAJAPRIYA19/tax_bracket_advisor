# app.py
from flask import Flask, render_template, request
from pyswip import Prolog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, threading

app = Flask(__name__)
app.secret_key = "replace-me-with-a-secret"

# Prolog initialization
PROLOG_FILE = "tax_advisor2.pl"
prolog = Prolog()
prolog.consult(PROLOG_FILE)

# Simple lock to protect Prolog interactions (pyswip & SWI-Prolog are not fully thread-safe)
prolog_lock = threading.Lock()

def get_slab_from_prolog(income):
    """Query Prolog to get the slab for given income."""
    query = f"get_slab({income}, Slab)."
    res = list(prolog.query(query, maxresult=1))
    if res:
        slab = res[0]["Slab"]

        # Case 1: slab is a Prolog term tax("..","..")
        if hasattr(slab, "args") and len(slab.args) == 2:
            part1 = slab.args[0]
            part2 = slab.args[1]
            # decode if bytes
            if isinstance(part1, bytes):
                part1 = part1.decode()
            if isinstance(part2, bytes):
                part2 = part2.decode()
            return f"{part1} – {part2}"

        # Case 2: slab already comes back as a tuple/list
        if isinstance(slab, (tuple, list)) and len(slab) == 2:
            part1, part2 = slab
            return f"{part1.decode() if isinstance(part1, bytes) else part1} – {part2.decode() if isinstance(part2, bytes) else part2}"

        # Fallback
        return str(slab)

    return "Unknown"


def run_prolog_clear_and_assert(age_val, income_val, deductions_dict):
    """
    Clear previous user facts and assert new ones for age, income and deductions.
    """
    with prolog_lock:
        # clear previous facts
        list(prolog.query("retractall(income(_))"))
        list(prolog.query("retractall(age(_))"))
        list(prolog.query("retractall(deduction(_, _))"))
        list(prolog.query("retractall(old_regime_tax(_))"))
        list(prolog.query("retractall(new_regime_tax(_))"))

        # assert current facts
        prolog.assertz(f"age({age_val})")
        prolog.assertz(f"income({income_val})")
        for sec, amt in deductions_dict.items():
            # ensure section name matches Prolog facts (quote it)
            prolog.assertz(f"deduction('{sec}', {amt})")

def call_tax_summary_and_collect():
    """
    Run tax_summary/0 and collect values we need from Prolog.
    Returns a dict of values.
    """
    with prolog_lock:
        # run the main prolog summary which also asserts old_regime_tax/1 and new_regime_tax/1
        list(prolog.query("tax_summary."))

        # read old and new tax (dynamic predicates asserted by tax_summary)
        old_tax_q = list(prolog.query("old_regime_tax(T)"))
        new_tax_q = list(prolog.query("new_regime_tax(T)"))
        old_tax = float(old_tax_q[0]['T']) if old_tax_q else 0.0
        new_tax = float(new_tax_q[0]['T']) if new_tax_q else 0.0

        # total deductions (Prolog predicate)
        td_q = list(prolog.query("total_deductions(TD)"))
        total_deductions = int(td_q[0]['TD']) if td_q else None

        # taxable incomes (old and new)
        ti_old_q = list(prolog.query("taxable_income(old, TIold)"))
        ti_new_q = list(prolog.query("taxable_income(new, TInew)"))
        ti_old = int(ti_old_q[0]['TIold']) if ti_old_q else None
        ti_new = int(ti_new_q[0]['TInew']) if ti_new_q else None

        # suggested regime
        sr_q = list(prolog.query("suggest_regime(R, OT, NT)"))
        suggested = sr_q[0]['R'] if sr_q else None

        # deduction tips (multiple solutions)
        tips = []
        for sol in prolog.query("deduction_gap(Sec, Gap)"):
            tips.append((sol['Sec'], int(sol['Gap'])))

    return {
        "old_tax": old_tax,
        "new_tax": new_tax,
        "total_deductions": total_deductions,
        "ti_old": ti_old,
        "ti_new": ti_new,
        "suggested_regime": suggested,
        "tips": tips
    }

def make_chart_base64(old_tax, new_tax):
    """Return a PNG image as base64 string comparing old vs new taxes."""
    regimes = ["Old Regime", "New Regime"]
    taxes = [old_tax, new_tax]
    colors = ['#1976D2', '#FF8F00']  # blue / orange

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
    bars = ax.bar(regimes, taxes, color=colors, edgecolor='black', linewidth=1)

    # highlight cheaper regime
    cheaper_index = int(taxes.index(min(taxes)))
    bars[cheaper_index].set_edgecolor('gold')
    bars[cheaper_index].set_linewidth(3)

    # value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(1, max(taxes)*0.01),
                f"₹{h:,.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # zooming to emphasize small differences:
    min_val = min(taxes)
    max_val = max(taxes)
    if max_val > 0:
        margin = max((max_val - min_val) * 0.25, max_val * 0.02)
        lower = max(min_val - margin, 0)
        upper = max_val + margin
        ax.set_ylim(lower, upper)

    ax.set_ylabel("Tax Payable (₹)")
    ax.set_title("Old vs New Regime Tax Comparison")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return img_b64

# Route handlers
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # read basic fields
        age = int(request.form.get('age', '0') or 0)
        income = int(request.form.get('income', '0') or 0)

        # Collect deduction fields from the form. Map form names -> Prolog section names
        deductions = {}
        # form field names from your index.html:
        # "80C", "EPF", "80D", "80CCD_1B", "LIFE"
        form_mapping = {
            "80C": "80C",
            "EPF": "EPF",
            "80D": "80D",
            "80CCD_1B": "80CCD(1B)",
            "LIFE": "Life Insurance"
        }
        for form_field, section_name in form_mapping.items():
            v = request.form.get(form_field)
            if v is None or v == "":
                continue
            try:
                amt = int(v)
                if amt > 0:
                    deductions[section_name] = amt
            except ValueError:
                # ignore invalid numbers, or you can flash an error
                pass

        # Tell Prolog the facts
        run_prolog_clear_and_assert(age, income, deductions)

        # Run tax logic and collect outputs
        prolog_results = call_tax_summary_and_collect()

        old_tax = prolog_results['old_tax']
        new_tax = prolog_results['new_tax']
        total_deductions = prolog_results['total_deductions']
        ti_old = prolog_results['ti_old']
        ti_new = prolog_results['ti_new']
        suggested = prolog_results['suggested_regime']
        tips = prolog_results['tips']

        # Get slab info for current income
        slab_info = get_slab_from_prolog(income)


        # make chart as base64 PNG
        chart_png = make_chart_base64(old_tax, new_tax)

        # prepare data for template
        return render_template(
            'result.html',
            income=income,
            age=age,
            deductions=deductions,
            total_deductions=total_deductions,
            ti_old=ti_old,
            ti_new=ti_new,
            old_tax=old_tax,
            new_tax=new_tax,
            suggested_regime=suggested,
            tips=tips,
            chart_png=chart_png,
            slab_info=slab_info
        )

    except Exception as e:
        # On error, show index with message
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)