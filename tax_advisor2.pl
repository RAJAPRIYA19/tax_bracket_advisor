% -----------------------------
% Dynamic predicates
:- dynamic tax_slab/3.
:- dynamic income/1.
:- dynamic age/1.
:- dynamic regime/1.
:- dynamic deduction/2.
:- dynamic old_regime_tax/1.
:- dynamic new_regime_tax/1.

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
surcharge(_, 5000000, 0.0).        
surcharge(_, 10000000, 0.10).     
surcharge(_, 20000000, 0.15).  
surcharge(old, 50000000, 0.25).    
surcharge(new, 999999999, 0.25). 
surcharge(old,999999999, 0.37).  

% -----------------------------
% Max deduction limits
max_deduction('80C', 150000).
max_deduction('80D', Max) :- age(A), (A >= 60 -> Max = 50000 ; Max = 25000).
max_deduction('80CCD(1B)', 50000).
max_deduction('EPF', 150000).
max_deduction('Life Insurance', 150000).
max_deduction('Standard', 50000).
max_deduction('NPS', 50000).

% -----------------------------
% Utility Rules

% Get tax slab structure based on income
get_slab(Income, tax("0–2.5L", "0%")) :-
    Income =< 250000, !.

get_slab(Income, tax("2.5L–5L", "5%")) :-
    Income =< 500000, !.

get_slab(Income, tax("5L–10L", "20%")) :-
    Income =< 1000000, !.

get_slab(_, tax("10L+", "30%")).

unify_example :-
    X = tax(slab, rate),
    writeln(X).

sum_list_custom([], 0).
sum_list_custom([H|T], Sum) :-
    sum_list_custom(T, Rest),
    Sum is H + Rest.

not_claimed(Section) :-
    \+ deduction(Section, _).

first_tax_slab(Regime, Limit, Rate) :-
    tax_slab(Regime, Limit, Rate), !.

ask_user_section :-
    write("Enter a section to check: "),
    read(Sec),
    ( deduction(Sec, Amt) -> format("Claimed amount: ₹~w~n", [Amt])
    ; writeln("Not claimed.") ).

show_text_chart(OldTax, NewTax) :-
    writeln("📊 [Chart will be displayed in Python]").

age_based_80d_limit(Limit) :-
    age(A),
    ( A >= 60 -> Limit = 50000 ; Limit = 25000 ).

total_deductions(Sum) :-
    findall(A, deduction(_, A), L),
    sum_list_custom(L, Partial),
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
    TI is I - 50000.

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
%surcharge application
apply_surcharge(Regime, BaseTax, FinalTax) :-
    income(I),
    findall((Limit, Rate), surcharge(Regime, Limit, Rate), Slabs),
    sort(Slabs, Sorted),
    find_surcharge_rate(I, Sorted, 0, SurchargeRate),
    FinalTax is BaseTax * (1 + SurchargeRate).

find_surcharge_rate(Income, [(Limit, Rate)|Rest], CurrentRate, FinalRate) :-
    ( Income =< Limit ->
        FinalRate = Rate
    ;
        find_surcharge_rate(Income, Rest, Rate, FinalRate)
    ).
find_surcharge_rate(_, [], Rate, Rate).

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

deduction_gap('80C', Gap) :-
    findall(Amount,
        (member(Section, ['80C', 'EPF', 'LifeInsurance', 'NPS']),
         deduction(Section, Amount)),
        Amounts),
    sum_list_custom(Amounts, Total),
    Gap is 150000 - Total,
    Gap > 0.

deduction_gap('80D', Gap) :-
    (deduction('80D', Used) -> true ; Used = 0),
    max_deduction('80D', Max),
    Gap is Max - Used,
    Gap > 0.

deduction_gap(S, Gap) :-
    \+ member(S, ['80C', 'EPF', 'LifeInsurance', 'NPS', '80D']),
    (deduction(S, Used) -> true ; Used = 0),
    max_deduction(S, Max),
    Gap is Max - Used,
    Gap > 0.

% -----------------------------
% Deduction Tips

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
    format("📌 The New Regime is suggested as it saves ₹~2f more tax than the Old Regime. You claimed deductions of ₹~w.~n", [Diff, Deductions]).

explain_choice(OldTax, NewTax, Deductions) :-
    Diff is NewTax - OldTax,
    Diff >= 0,
    format("📌 The Old Regime is suggested as it saves ₹~2f more tax due to your total deductions of ₹~w.~n", [Diff, Deductions]).

% -----------------------------
% Final Tax Summary

tax_summary :-
    unify_example,
    income(Income),
    age(Age),
    total_deductions(TotalDeduction),
    compute_tax(old, OldTax),
    compute_tax(new, NewTax),

    % Store computed tax values for Python
    retractall(old_regime_tax(_)),
    retractall(new_regime_tax(_)),
    assertz(old_regime_tax(OldTax)),
    assertz(new_regime_tax(NewTax)),

    suggest_regime(BestRegime, OldTax, NewTax),
    taxable_income(BestRegime, TaxableIncome),
    format("Tax Analysis Suggestion:~n~n"),
    format("Income: ₹~w~n", [Income]),
    format("Age: ~w~n", [Age]),
    format("Deductions (incl. ₹50,000 standard): ₹~w~n", [TotalDeduction]),
    format("Taxable Income: ₹~w~n", [TaxableIncome]),
    format("Old Regime Tax: ₹~2f~n", [OldTax]),
    format("New Regime Tax: ₹~2f~n", [NewTax]),
    format("Suggested Regime: ~w~n", [BestRegime]),
    explain_choice(OldTax, NewTax, TotalDeduction),
    print_deduction_tips,
    show_text_chart(OldTax, NewTax).
