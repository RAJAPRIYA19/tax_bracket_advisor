# app.py
from flask import Flask, render_template, request
from pyswip import Prolog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, threading, re

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


def parse_plan_obj(raw_plan):
    """
    Normalize Prolog deduction terms into clean Python tuples.
    """
    result = []

    def parse_element(elem):
        try:
            if hasattr(elem, "args") and len(elem.args) == 2:
                sec = ensure_str(elem.args[0]).strip("'\"")
                amt = int(str(elem.args[1]))
                return (sec, amt)
        except Exception:
            pass

        try:
            if isinstance(elem, (tuple, list)) and len(elem) == 2:
                sec = ensure_str(elem[0]).strip("'\"")
                amt = int(elem[1])
                return (sec, amt)
        except Exception:
            pass

        try:
            s = ensure_str(elem)
            m = re.match(r"deduction\(\s*'?([A-Za-z0-9\(\)\/ ]+)'?\s*,\s*(\d+)\s*\)", s)
            if m:
                return (m.group(1).strip(), int(m.group(2)))
        except Exception:
            pass

        return None  # instead of returning (string,0)

    # Case A: list/tuple
    if isinstance(raw_plan, (list, tuple)):
        for e in raw_plan:
            parsed = parse_element(e)
            if parsed:
                result.append(parsed)
        return result

    # Case B: pyswip wrapper
    if hasattr(raw_plan, "value") and isinstance(raw_plan.value, (list, tuple)):
        for e in raw_plan.value:
            parsed = parse_element(e)
            if parsed:
                result.append(parsed)
        return result

    # Case C: raw string fallback
    try:
        s = ensure_str(raw_plan)
        items = re.findall(r"deduction\(\s*'?([A-Za-z0-9\(\)\/ ]+)'?\s*,\s*(\d+)\s*\)", s)
        for sec, amt in items:
            result.append((sec.strip(), int(amt)))
        if result:
            return result
    except Exception:
        pass

    return result




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
            # escape single quotes inside sec if any (unlikely)
            sec_escaped = sec.replace("'", "\\'")
            prolog.assertz(f"deduction('{sec_escaped}', {amt})")

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
        total_deductions = int(td_q[0]['TD']) if td_q else 0

        # taxable incomes
        ti_old_q = list(prolog.query("taxable_income(old, TIold)"))
        ti_new_q = list(prolog.query("taxable_income(new, TInew)"))
        ti_old = int(ti_old_q[0]['TIold']) if ti_old_q else 0
        ti_new = int(ti_new_q[0]['TInew']) if ti_new_q else 0

        # suggested regime
        sr_q = list(prolog.query("suggest_regime(R, OT, NT)"))
        suggested = ensure_str(sr_q[0]['R']) if sr_q else None

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

def run_optimize(algo_atom):
    """
    Calls Prolog optimize(algo, Plan, Tax). Returns:
      plan_list: [(section, amount), ...] or [] if none
      tax: float or None
    algo_atom should be 'astar' or 'ao' (string)
    """
    prog = f"optimize({algo_atom}, Plan, Tax)."
    with prolog_lock:
        res = list(prolog.query(prog, maxresult=1))
    if not res:
        return [], None, None  # no plan
    binding = res[0]
    raw_plan = binding.get("Plan")
    raw_tax = binding.get("Tax")
    plan_list = parse_plan_obj(raw_plan)
    try:
        tax_val = float(raw_tax) if raw_tax is not None else None
    except Exception:
        tax_val = None
    # Also return raw_plan for debugging / advanced displays
    return plan_list, tax_val, raw_plan

def make_chart_base64(old_tax, new_tax, optimized_tax=None):
    """Make bar chart for Old vs New regime taxes and optimized (optional), return as base64 string."""
    labels = ["Old Regime", "New Regime"]
    values = [old_tax, new_tax]
    if optimized_tax is not None:
        labels.append("Optimized (chosen)")
        values.append(optimized_tax)

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
    bars = ax.bar(labels, values)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + max(1, max(values) * 0.01),
                f"₹{h:,.0f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel("Tax Payable (₹)")
    ax.set_title("Regime / Optimized Tax Comparison")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

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

        # Run both optimizers
        plan_astar, tax_astar, raw_astar = run_optimize("astar")
        plan_ao, tax_ao, raw_ao = run_optimize("ao")

        # Build explanations for AO* (based on rules you have in Prolog)
        ao_explanation = []
        # compute initial totals for strict NPS check and conditions using the same logic as Prolog
        init_total = sum(deductions.get(k, 0) for k in ["80C", "80D", "Life Insurance", "EPF"])
        # check presence in plan
        ao_sections = {sec for sec, _ in plan_ao}
        # NPS strict dependency
        if "80CCD(1B)" not in ao_sections and init_total < 50000:
            ao_explanation.append("NPS (80CCD(1B)) was excluded because your initial combined 80C+80D+Life+EPF < ₹50,000.")
        # 80D sequential condition: full 80D allowed only if 80C+Life+EPF >= 150000 (Prolog rule)
        if "80D" not in ao_sections:
            # check whether AO* considered adding 80D - if not present, say why if 80C not maxed
            max80c = 150000
            # find optimized 80C in AO plan
            ao_80c_amt = next((amt for sec, amt in plan_ao if sec == "80C"), 0)
            # also consider user-provided 80C (deductions)
            current_80c = deductions.get("80C", 0)
            total_80c = max(ao_80c_amt, current_80c)
            if total_80c < max80c:
                ao_explanation.append("80D was not added at full limit because 80C + Life Insurance + EPF were not maxed to ₹1.5L.")

        # A* explanation (simple)
        astar_explanation = []
        if not plan_astar:
            astar_explanation.append("A* did not find any improvement from your current deductions with the configured step sizes.")
        else:
            astar_explanation.append("A* used incremental search with heuristic = 5% of income to find the lowest-tax plan.")

        # Choose which optimized-tax to show on chart: pick the smaller of A* and AO* (if both exist)
        optimized_tax_for_chart = None
        if tax_astar is not None and tax_ao is not None:
            optimized_tax_for_chart = min(tax for tax in [tax_astar, tax_ao] if tax is not None)
        elif tax_astar is not None:
            optimized_tax_for_chart = tax_astar
        elif tax_ao is not None:
            optimized_tax_for_chart = tax_ao

        chart_png = make_chart_base64(results['old_tax'], results['new_tax'], optimized_tax_for_chart)
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
            tips=results['tips'],
            unused=results['not_claimed'],
            chart_png=chart_png,
            slab_info=slab_info,
            # Optimizer outputs
            plan_astar=plan_astar, tax_astar=tax_astar, raw_astar=raw_astar,
            plan_ao=plan_ao, tax_ao=tax_ao, raw_ao=raw_ao,
            ao_explanation=ao_explanation, astar_explanation=astar_explanation
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
