% -----------------------------
% Dynamic predicates
:- dynamic tax_slab/3.
:- dynamic income/1.
:- dynamic age/1.
:- dynamic regime/1.
:- dynamic deduction/2.

% -----------------------------
% Tax slabs FY 2024–25

% Old Regime
tax_slab(old, 250000, 0).
tax_slab(old, 500000, 0.05).
tax_slab(old, 1000000, 0.2).
tax_slab(old, 5000000, 0.3).
tax_slab(old, 99999999, 0.3).  % surcharge applies separately

% New Regime
tax_slab(new, 300000, 0).
tax_slab(new, 700000, 0.05).
tax_slab(new, 1000000, 0.1).
tax_slab(new, 1200000, 0.15).
tax_slab(new, 1500000, 0.2).
tax_slab(new, 5000000, 0.3).
tax_slab(new, 99999999, 0.3).  % surcharge applies separately

% -----------------------------
% Surcharge slabs
% Format: surcharge(Regime, IncomeThreshold, Rate)

surcharge(_, 50000000, 0.1).         % ₹50L–₹1Cr
surcharge(_, 100000000, 0.15).       % ₹1Cr–₹2Cr
surcharge(old, 200000000, 0.25).     % ₹2Cr–₹5Cr (only old)
surcharge(old, 999999999, 0.37).     % ₹5Cr+ (only old)
surcharge(new, 999999999, 0.15).     % capped for new

% -----------------------------
% Max deduction limits (all section names quoted and upper-case to match Python)
max_deduction('80C', 150000).
max_deduction('80D', Max) :- age(A), (A >= 60 -> Max = 50000 ; Max = 25000).
max_deduction('80CCD(1B)', 50000).
max_deduction('EPF', 150000).
max_deduction('Life Insurance', 150000).
max_deduction('Standard', 50000).
max_deduction('NPS', 50000).  % optional alias

% -----------------------------
% Utility Rules

print_max_deduction_limits :-
    writeln("🔢 Maximum Deduction Limits (FY 2024–25):"),
    age_based_80d_limit(L),
    format(" - Medical Insurance (80D): ₹~w~n", [L]),
    format(" - Life Insurance (80C): ₹150000~n"),
    format(" - EPF (80C): ₹150000~n"),
    format(" - NPS (80CCD(1B)): ₹50000~n"),
    format(" - Standard Deduction: ₹50000 (auto applied)~n~n").

age_based_80d_limit(Limit) :-
    age(A),
    ( A >= 60 -> Limit = 50000 ; Limit = 25000 ).

total_deductions(Sum) :-
    findall(A, deduction(_, A), L),
    sum_list(L, Partial),
    max_deduction('Standard', Std),
    Sum is Partial + Std.

age_based_exemption(E) :-
    age(A),
    (A >= 60 -> E is 300000 ; E is 250000).

get_sorted_slabs(Regime, Sorted) :-
    findall((Limit, Rate), tax_slab(Regime, Limit, Rate), L),
    sort(L, Sorted).

% -----------------------------
% Taxable Income

taxable_income(old, TI) :-
    income(I),
    total_deductions(D),
    age_based_exemption(E),
    Raw is I - D - E,
    TI is max(0, Raw).

taxable_income(new, TI) :-
    income(I),
    TI is I - 50000.  % Standard deduction only

% -----------------------------
% Progressive Tax Computation

progressive_tax(_, _, [], _, 0).
progressive_tax(Income, PrevLimit, [(Limit, Rate)|Rest], Acc, Tax) :-
    Income =< Limit,
    Portion is Income - PrevLimit,
    Temp is Portion * Rate,
    Tax is Acc + Temp.

progressive_tax(Income, PrevLimit, [(Limit, Rate)|Rest], Acc, Tax) :-
    Income > Limit,
    Portion is Limit - PrevLimit,
    Temp is Portion * Rate,
    NewAcc is Acc + Temp,
    progressive_tax(Income, Limit, Rest, NewAcc, Tax).

% -----------------------------
% Surcharge Application

apply_surcharge(Regime, BaseTax, FinalTax) :-
    income(I),
    findall((Limit, Rate), surcharge(Regime, Limit, Rate), Slabs),
    sort(Slabs, Sorted),
    find_surcharge_rate(I, Sorted, 0, SurchargeRate),
    FinalTax is BaseTax * (1 + SurchargeRate).

find_surcharge_rate(_, [], Rate, Rate).
find_surcharge_rate(I, [(Limit, Rate)|_], _, Rate) :- I =< Limit, !.
find_surcharge_rate(I, [_|Rest], _, Final) :- find_surcharge_rate(I, Rest, _, Final).

% -----------------------------
% Compute Final Tax

compute_tax(Regime, FinalTax) :-
    taxable_income(Regime, TI),
    get_sorted_slabs(Regime, Slabs),
    progressive_tax(TI, 0, Slabs, 0, BaseTax),
    apply_surcharge(Regime, BaseTax, FinalTax).

% -----------------------------
% Regime Suggestion

suggest_regime(BestRegime, OldTax, NewTax) :-
    compute_tax(old, OldTax),
    compute_tax(new, NewTax),
    (OldTax < NewTax -> BestRegime = old ; BestRegime = new).

% -----------------------------
% Deduction Gap Logic

% 80C combined cap
deduction_gap('80C', Gap) :-
    findall(Amount,
        (member(Section, ['80C', 'EPF', 'LifeInsurance', 'NPS']),
         deduction(Section, Amount)),
        Amounts),
    sum_list(Amounts, Total),
    Gap is 150000 - Total,
    Gap > 0.

% 80D logic based on age
deduction_gap('80D', Gap) :-
    (deduction('80D', Used) -> true ; Used = 0),
    max_deduction('80D', Max),
    Gap is Max - Used,
    Gap > 0.

% Other deductions (non-80C, non-80D)
deduction_gap(S, Gap) :-
    \+ member(S, ['80C', 'EPF', 'LifeInsurance', 'NPS', '80D']),
    (deduction(S, Used) -> true ; Used = 0),
    max_deduction(S, Max),
    Gap is Max - Used,
    Gap > 0.
% -----------------------------
% Deduction Tips (including 80C)

print_deduction_tips :-
    deduction_gap(S, G),
    format("💡 Tip: Invest ₹~w more in ~w to save tax.~n", [G, S]),
    fail.
print_deduction_tips.

% -----------------------------
% Regime Explanation

explain_choice(OldTax, NewTax, Deductions) :-
    Diff is OldTax - NewTax,
    Diff > 0,
    format("📌 The New Regime is suggested as it saves ₹~2f more tax than the Old Regime. You claimed deductions of ₹~w, which may not be fully beneficial in the Old Regime.~n", [Diff, Deductions]).

explain_choice(OldTax, NewTax, Deductions) :-
    Diff is NewTax - OldTax,
    Diff >= 0,
    format("📌 The Old Regime is suggested as it saves ₹~2f more tax due to your total deductions of ₹~w.~n", [Diff, Deductions]).

% -----------------------------
% Final Tax Summary

tax_summary :-
    income(Income),
    age(Age),
    total_deductions(TotalDeduction),
    compute_tax(old, OldTax),
    compute_tax(new, NewTax),
    suggest_regime(BestRegime, OldTax, NewTax),
    taxable_income(BestRegime, TaxableIncome),
    format("📊 Tax Analysis Suggestion:~n~n"),
    format("🪙 Income: ₹~w~n", [Income]),
    format("👤 Age: ~w~n", [Age]),
    format("📉 Deductions (incl. ₹50,000 standard): ₹~w~n", [TotalDeduction]),
    format("💰 Taxable Income: ₹~w~n", [TaxableIncome]),
    format("🧮 Old Regime Tax (w/ surcharge): ₹~2f~n", [OldTax]),
    format("🧮 New Regime Tax (w/ surcharge): ₹~2f~n", [NewTax]),
    format("🎯 Suggested Regime: ~w~n", [BestRegime]),
    explain_choice(OldTax, NewTax, TotalDeduction),
    print_deduction_tips.
