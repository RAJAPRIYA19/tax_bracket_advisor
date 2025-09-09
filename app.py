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

# Lock for Prolog (not fully thread-safe)
prolog_lock = threading.Lock()


# -------------------------------
# Utility functions
# -------------------------------
def ensure_str(x):
    """Convert Prolog atoms or bytes to Python string."""
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def get_slab_from_prolog(income):
    """Query Prolog get_slab/2 to retrieve the applicable tax slab."""
    with prolog_lock:
        res = list(prolog.query(f"get_slab({income}, Slab).", maxresult=1))
    if res:
        slab = res[0]["Slab"]

        # Case 1: Prolog compound term tax("..","..")
        if hasattr(slab, "args") and len(slab.args) == 2:
            part1, part2 = map(ensure_str, slab.args)
            return f"{part1} – {part2}"

        # Case 2: Already Python tuple/list
        if isinstance(slab, (tuple, list)) and len(slab) == 2:
            part1, part2 = map(ensure_str, slab)
            return f"{part1} – {part2}"

        # Fallback
        return ensure_str(slab)

    return "Unknown"


def run_prolog_clear_and_assert(age_val, income_val, deductions_dict):
    """
    Clear previous user facts and assert new ones (age, income, deductions).
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
            prolog.assertz(f"deduction('{sec}', {amt})")


def call_tax_summary_and_collect():
    """
    Call tax_summary/0 and collect results: tax values, deductions, regime, tips, not_claimed.
    """
    with prolog_lock:
        list(prolog.query("tax_summary."))  # runs summary & asserts dynamic facts

        # taxes
        old_tax_q = list(prolog.query("old_regime_tax(T)"))
        new_tax_q = list(prolog.query("new_regime_tax(T)"))
        old_tax = float(old_tax_q[0]['T']) if old_tax_q else 0.0
        new_tax = float(new_tax_q[0]['T']) if new_tax_q else 0.0

        # deductions total
        td_q = list(prolog.query("total_deductions(TD)"))
        total_deductions = int(td_q[0]['TD']) if td_q else None

        # taxable incomes
        ti_old_q = list(prolog.query("taxable_income(old, TIold)"))
        ti_new_q = list(prolog.query("taxable_income(new, TInew)"))
        ti_old = int(ti_old_q[0]['TIold']) if ti_old_q else None
        ti_new = int(ti_new_q[0]['TInew']) if ti_new_q else None

        # suggested regime
        sr_q = list(prolog.query("suggest_regime(R, OT, NT)"))
        suggested = sr_q[0]['R'] if sr_q else None

        # deduction tips (deduction_gap/2)
        tips = []
        for sol in prolog.query("deduction_gap(Sec, Gap)"):
            tips.append((ensure_str(sol['Sec']), int(sol['Gap'])))

        # not_claimed sections
        not_claimed_list = []
        for sol in prolog.query("not_claimed(S)"):
            not_claimed_list.append(ensure_str(sol['S']))

    return {
        "old_tax": old_tax,
        "new_tax": new_tax,
        "total_deductions": total_deductions,
        "ti_old": ti_old,
        "ti_new": ti_new,
        "suggested_regime": suggested,
        "tips": tips,
        "not_claimed": not_claimed_list
    }


def make_chart_base64(old_tax, new_tax):
    """Make bar chart for Old vs New regime taxes, return as base64 string."""
    regimes = ["Old Regime", "New Regime"]
    taxes = [old_tax, new_tax]
    colors = ['#1976D2', '#FF8F00']

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
    bars = ax.bar(regimes, taxes, color=colors, edgecolor='black', linewidth=1)

    # highlight cheaper
    cheaper_index = int(taxes.index(min(taxes)))
    bars[cheaper_index].set_edgecolor('gold')
    bars[cheaper_index].set_linewidth(3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + max(1, max(taxes) * 0.01),
                f"₹{h:,.0f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

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
    return base64.b64encode(buf.read()).decode('ascii')


# -------------------------------
# Flask routes
# -------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    try:
        age = int(request.form.get('age', '0') or 0)
        income = int(request.form.get('income', '0') or 0)

        # Map form fields → Prolog deduction sections
        deductions = {}
        form_mapping = {
            "80C": "80C",
            "EPF": "EPF",
            "80D": "80D",
            "80CCD_1B": "80CCD(1B)",
            "LIFE": "Life Insurance"
        }
        
        for form_field, section_name in form_mapping.items():
            v = request.form.get(form_field)
            if v:
                try:
                    amt = int(v)
                    if amt > 0:
                        deductions[section_name] = amt
                except ValueError:
                    pass  # ignore invalid numbers

        # Update Prolog facts
        run_prolog_clear_and_assert(age, income, deductions)

        # Collect results from Prolog
        results = call_tax_summary_and_collect()

        tips = []
        for sol in prolog.query("deduction_gap(S, Gap)"):
            s = sol["S"]
            gap = sol["Gap"]
            if gap > 0:
                # store as tuple instead of string
                tips.append((s, gap))

        not_claimed = []
        for sol in prolog.query("not_claimed(S)"):
            not_claimed.append(sol["S"])


        # Chart + slab info
        chart_png = make_chart_base64(results['old_tax'], results['new_tax'])
        slab_info = get_slab_from_prolog(income)

        return render_template(
            'result.html',
            income=income,
            age=age,
            deductions=deductions,
            total_deductions=results['total_deductions'],
            ti_old=results['ti_old'],
            ti_new=results['ti_new'],
            old_tax=results['old_tax'],
            new_tax=results['new_tax'],
            suggested_regime=results['suggested_regime'],
            tips=tips,
            unused=not_claimed,
            chart_png=chart_png,
            slab_info=slab_info
        )
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
