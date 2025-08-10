% -----------------------------
% Dynamic predicates
:- dynamic tax_slab/3.
:- dynamic user/3.
:- dynamic income/2.
:- dynamic deductions/3.

% -----------------------------
% Sample Facts

% user(Name, Age, Regime).
user(ram, 30, old).
user(sita, 26, new).
user(kumar, 67, old).

% income(Name, AnnualIncome).
income(ram, 750000).
income(sita, 850000).
income(kumar, 600000).

% deductions(Name, Section, Amount).
deductions(ram, '80C', 120000).
deductions(ram, '80D', 30000).
deductions(kumar, '80C', 100000).
deductions(sita, '80C', 0).  % new regime, no deduction

% tax_slab(Regime, UpperLimit, Rate).
% Old Regime
tax_slab(old, 250000, 0).
tax_slab(old, 500000, 0.05).
tax_slab(old, 1000000, 0.2).
tax_slab(old, 9999999, 0.3).

% New Regime
tax_slab(new, 300000, 0).
tax_slab(new, 600000, 0.05).
tax_slab(new, 900000, 0.1).
tax_slab(new, 1200000, 0.15).
tax_slab(new, 1500000, 0.2).
tax_slab(new, 9999999, 0.3).

% -----------------------------
% Utility Rules

% Sum of deductions for a user
total_deductions(User, Total) :-
    findall(Amount, deductions(User, _, Amount), List),
    sum_list(List, Total).

% Age-based exemptions (for old regime only)
age_based_exemption(Age, Exemption) :-
    Age >= 60, Exemption is 300000.
age_based_exemption(Age, Exemption) :-
    Age < 60, Exemption is 250000.

% Max allowed deduction limits
max_deduction('80C', 150000).
max_deduction('80D', 50000).

% Sorted slabs for progressive calculation
get_sorted_slabs(Regime, Sorted) :-
    findall((Limit, Rate), tax_slab(Regime, Limit, Rate), List),
    sort(List, Sorted).

% -----------------------------
% Core Rule: Taxable Income

% Taxable income after deductions and exemptions
taxable_income(User, Taxable) :-
    income(User, Income),
    user(User, Age, Regime),
    (
        Regime = old ->
            total_deductions(User, Deducted),
            age_based_exemption(Age, Exemption),
            Raw is Income - Deducted - Exemption,
            Taxable is max(0, Raw)
        ;
        Taxable = Income
    ).

% -----------------------------
% Progressive Tax Calculation

% Tax calculated slab by slab
progressive_tax(_, _, [], _, 0).

progressive_tax(Income, PrevLimit, [(Limit, Rate)|Rest], Acc, Tax) :-
    Income =< Limit,
    Portion is Income - PrevLimit,
    TempTax is Portion * Rate,
    Tax is Acc + TempTax.

progressive_tax(Income, PrevLimit, [(Limit, Rate)|Rest], Acc, Tax) :-
    Income > Limit,
    Portion is Limit - PrevLimit,
    TempTax is Portion * Rate,
    NewAcc is Acc + TempTax,
    progressive_tax(Income, Limit, Rest, NewAcc, Tax).

% Final rule to compute total tax
compute_progressive_tax(User, Tax) :-
    taxable_income(User, Taxable),
    user(User, _, Regime),
    get_sorted_slabs(Regime, Slabs),
    progressive_tax(Taxable, 0, Slabs, 0, Tax).

% -----------------------------
% Regime Suggestion

suggest_regime(User) :-
    income(User, Income),
    total_deductions(User, Deducted),
    user(User, Age, _),

    % old regime
    age_based_exemption(Age, Exempt),
    OldTaxable is Income - Deducted - Exempt,
    get_sorted_slabs(old, OldSlabs),
    progressive_tax(OldTaxable, 0, OldSlabs, 0, OldTax),

    % new regime
    get_sorted_slabs(new, NewSlabs),
    progressive_tax(Income, 0, NewSlabs, 0, NewTax),

    (
        OldTax < NewTax ->
        format("~w should choose Old Regime. Tax: ₹~2f~n", [User, OldTax])
        ;
        format("~w should choose New Regime. Tax: ₹~2f~n", [User, NewTax])
    ).

% -----------------------------
% Quantifier Examples

% List all users
list_users :-
    forall(user(Name, _, _), writeln(Name)).

% List user tax amounts
list_user_taxes :-
    forall(user(Name, _, _),
        ( compute_progressive_tax(Name, Tax),
          format("~w pays ₹~2f tax~n", [Name, Tax])
        )
    ).

% List unused deduction space
deduction_gap(User, Section, Gap) :-
    deductions(User, Section, Used),
    max_deduction(Section, Max),
    Gap is Max - Used,
    Gap > 0.

show_all_gaps :-
    forall(deduction_gap(User, Section, Gap),
        format("~w can still use ₹~w under ~w~n", [User, Gap, Section])
    ).

% -----------------------------
% Simulate Extra Deduction

simulate_extra(User, Section, Extra) :-
    deductions(User, Section, Used),
    max_deduction(Section, Max),
    Possible is min(Used + Extra, Max),
    Delta is Possible - Used,

    total_deductions(User, Current),
    NewDeducted is Current + Delta,

    user(User, Age, _),
    age_based_exemption(Age, Exempt),

    income(User, Income),
    NewTaxable is Income - NewDeducted - Exempt,
    get_sorted_slabs(old, Slabs),
    progressive_tax(NewTaxable, 0, Slabs, 0, NewTax),

    format("If ~w invests ₹~w more in ~w, new tax is ₹~2f~n", [User, Delta, Section, NewTax]).

simulate_multiple_investments(User, Section) :-
    between(0, 50000, Extra),
    simulate_extra(User, Section, Extra),
    fail.
simulate_multiple_investments(_, _).

% -----------------------------
% Explain Suggestion Reason

why_suggested(User) :-
    income(User, Income),
    total_deductions(User, Deducted),
    user(User, Age, _),
    age_based_exemption(Age, Exempt),

    OldTaxable is Income - Deducted - Exempt,
    get_sorted_slabs(old, OldSlabs),
    progressive_tax(OldTaxable, 0, OldSlabs, 0, OldTax),

    get_sorted_slabs(new, NewSlabs),
    progressive_tax(Income, 0, NewSlabs, 0, NewTax),

    format("User: ~w~nIncome: ₹~w~nDeductions: ₹~w~nExemption: ₹~w~n", [User, Income, Deducted, Exempt]),
    format("Old Regime Taxable: ₹~w~nTax: ₹~2f~n", [OldTaxable, OldTax]),
    format("New Regime Taxable: ₹~w~nTax: ₹~2f~n", [Income, NewTax]),

    (
        OldTax < NewTax ->
        writeln("Suggested: Old Regime (Lower Tax)")
        ;
        writeln("Suggested: New Regime (Lower Tax)")
    ).

% -----------------------------
% Check for Deduction Errors (evasion risk)

check_tax_evasion(User) :-
    deductions(User, Section, Amount),
    max_deduction(Section, Max),
    Amount > Max,
    format("⚠️ Warning: ~w claimed ₹~w in ~w (Limit: ₹~w)~n", [User, Amount, Section, Max]).

% -----------------------------
% Summary Report

user_summary(User) :-
    user(User, Age, Regime),
    income(User, Income),
    total_deductions(User, Deducted),
    taxable_income(User, Taxable),
    compute_progressive_tax(User, Tax),
    format("User: ~w~nAge: ~w~nRegime: ~w~nIncome: ₹~w~nDeductions: ₹~w~nTaxable: ₹~w~nTax: ₹~2f~n~n",
           [User, Age, Regime, Income, Deducted, Taxable, Tax]).
