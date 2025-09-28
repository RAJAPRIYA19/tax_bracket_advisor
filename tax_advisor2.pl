% ================================================================
% Tax Advisor Code with Old/New Bracket Fix + Export Both Brackets
% ================================================================

:- dynamic tax_slab/3.
:- dynamic income/1.
:- dynamic age/1.
:- dynamic regime/1.
:- dynamic deduction/2.
:- dynamic old_regime_tax/1.
:- dynamic new_regime_tax/1.
:- dynamic case/6.

:- use_module(library(heaps)).

% ================================================================
% Tax slabs FY 2024–25
% ================================================================

% Old Regime
tax_slab(old, 250000, 0).
tax_slab(old, 500000, 0.05).
tax_slab(old, 1000000, 0.2).
tax_slab(old, 5000000, 0.3).
tax_slab(old, 99999999, 0.3).

% New Regime
tax_slab(new, 300000, 0).
tax_slab(new, 700000, 0.05).
tax_slab(new, 1000000, 0.1).
tax_slab(new, 1200000, 0.15).
tax_slab(new, 1500000, 0.2).
tax_slab(new, 5000000, 0.3).
tax_slab(new, 99999999, 0.3).

% ================================================================
% Surcharge slabs
% ================================================================
surcharge(_, 5000000, 0.0).        
surcharge(_, 10000000, 0.10).     
surcharge(_, 20000000, 0.15).  
surcharge(old, 50000000, 0.25).    
surcharge(new, 999999999, 0.25). 
surcharge(old,999999999, 0.37).  

% ================================================================
% Max deduction limits
% ================================================================
max_deduction('80C', 150000).
max_deduction('80D', Max) :- age(A), (A >= 60 -> Max = 50000 ; Max = 25000).
max_deduction('80CCD(1B)', 50000).
max_deduction('EPF', 150000).
max_deduction('Life Insurance', 150000).
max_deduction('Standard', 50000).
max_deduction('NPS', 50000).

% ================================================================
% Utilities
% ================================================================
get_slab(Income, tax("0-2.5L", "0%")) :-
    Income =< 250000, !.
get_slab(Income, tax("2.5L-5L", "5%")) :-
    Income =< 500000, !.
get_slab(Income, tax("5L-10L", "20%")) :-
    Income =< 1000000, !.
get_slab(_, tax("10L+", "30%")).

total_income(Income) :- income(Income).

sum_list_custom([], 0).
sum_list_custom([H|T], Sum) :- sum_list_custom(T, Rest), Sum is H + Rest.

deduction_sections(['80C', '80D', '80CCD(1B)', 'EPF', 'NPS', 'Life Insurance']).

not_claimed(S) :- deduction_sections(All), member(S, All), \+ deduction(S, _).

suggest_unclaimed_deductions(Suggestions) :-
    findall(Section, (max_deduction(Section, _), not_claimed(Section)), Suggestions).

age_based_80d_limit(Limit) :- age(A), ( A >= 60 -> Limit = 50000 ; Limit = 25000 ).

total_deductions(Sum) :-
    findall(A, deduction(_, A), L),
    sum_list_custom(L, Partial),
    max_deduction('Standard', Std),
    Sum is Partial + Std.

age_based_exemption(E) :- age(A), (A >= 60 -> E is 300000 ; E is 250000).

get_sorted_slabs(Regime, Sorted) :-
    findall((Limit, Rate), tax_slab(Regime, Limit, Rate), L),
    sort(L, Sorted).

% ================================================================
% Taxable income
% ================================================================
taxable_income(old, TI) :-
    income(I), total_deductions(D), age_based_exemption(E),
    Raw is I - D - E, TI is max(0, Raw).

taxable_income(new, TI) :-
    income(I), TI is I - 50000.

% ================================================================
% Progressive tax
% ================================================================
progressive_tax(_, _, [], _, 0).
progressive_tax(Income, Prev, [(Limit, Rate)|Rest], Acc, Tax) :-
    Income =< Limit, Portion is Income - Prev, Temp is Portion * Rate, Tax is Acc + Temp.
progressive_tax(Income, Prev, [(Limit, Rate)|Rest], Acc, Tax) :-
    Income > Limit, Portion is Limit - Prev, Temp is Portion * Rate,
    NewAcc is Acc + Temp, progressive_tax(Income, Limit, Rest, NewAcc, Tax).

% ================================================================
% Surcharge
% ================================================================
apply_surcharge(Regime, BaseTax, FinalTax) :-
    income(I),
    findall((Limit, Rate), surcharge(Regime, Limit, Rate), Slabs),
    sort(Slabs, Sorted),
    find_surcharge_rate(I, Sorted, 0, SurchargeRate),
    FinalTax is BaseTax * (1 + SurchargeRate).

find_surcharge_rate(Income, [(Limit, Rate)|Rest], _, FinalRate) :-
    ( Income =< Limit -> FinalRate = Rate ; find_surcharge_rate(Income, Rest, Rate, FinalRate) ).
find_surcharge_rate(_, [], Rate, Rate).

% ================================================================
% Compute Tax
% ================================================================
compute_tax(Regime, FinalTax) :-
    taxable_income(Regime, TI),
    get_sorted_slabs(Regime, Slabs),
    progressive_tax(TI, 0, Slabs, 0, BaseTax),
    apply_surcharge(Regime, BaseTax, FinalTax).

% ================================================================
% Deduction Gaps
% ================================================================
deduction_gap('80C', Gap) :-
    findall(Amount, (member(S, ['80C','EPF','Life Insurance','NPS']), deduction(S, Amount)), Amounts),
    sum_list_custom(Amounts, Total),
    Gap is 150000 - Total, Gap > 0.

deduction_gap('80D', Gap) :-
    (deduction('80D', Used) -> true ; Used = 0),
    max_deduction('80D', Max),
    Gap is Max - Used, Gap > 0.

deduction_gap(S, Gap) :-
    \+ member(S, ['80C','EPF','Life Insurance','NPS','80D']),
    (deduction(S, Used) -> true ; Used = 0),
    max_deduction(S, Max),
    Gap is Max - Used, Gap > 0.

print_deduction_tips :- deduction_gap(S,G), format("💡 Tip: Invest ₹~w more in ~w to save tax.~n",[G,S]), fail.
print_deduction_tips.

% ================================================================
% Regime Suggestion
% ================================================================
suggest_regime(Best, OldTax, NewTax) :-
    compute_tax(old, OldTax), compute_tax(new, NewTax),
    (OldTax < NewTax -> Best = old ; Best = new).

% ================================================================
% Tax Summary
% ================================================================
tax_summary :-
    income(_), age(_), total_deductions(_),
    compute_tax(old, OldTax), compute_tax(new, NewTax),
    retractall(old_regime_tax(_)), retractall(new_regime_tax(_)),
    assertz(old_regime_tax(OldTax)), assertz(new_regime_tax(NewTax)),
    suggest_regime(_, OldTax, NewTax).

% ================================================================
% ================================================================
% A* Deduction Optimizer (Appended, New Feature)
% ================================================================
% ================================================================

% Collapse plan: keep max deduction per section
collapse_plan(Full, Collapsed) :-
    collapse_plan(Full, _{}, Dict),
    dict_pairs(Dict, _, Pairs),
    findall(deduction(S, Max), member(S-Max, Pairs), Collapsed).

collapse_plan([], Dict, Dict).
collapse_plan([deduction(S, A)|Rest], Dict0, Dict) :-
    ( get_dict(S, Dict0, Old) ->
        (A > Old -> put_dict(S, Dict0, A, Dict1) ; Dict1 = Dict0)
    ; put_dict(S, Dict0, A, Dict1) ),
    collapse_plan(Rest, Dict1, Dict).

% Step sizes (coarser for AO* speed)
step_size('80C', 50000).
step_size('80D', 25000).
step_size('80CCD(1B)', 25000).

% Section limits
section_limit('80C', 150000).
section_limit('80D', Limit) :- age(A), (A < 60 -> Limit = 25000 ; Limit = 50000).
section_limit('80CCD(1B)', 50000).

% Start state
start_state(state(0,0,0)).

% Apply deduction action
apply_action(state(C80C,C80D,NPS), deduction(Sec,NewVal), state(N80C,N80D,NNPS)) :-
    step_size(Sec, Step),
    ( Sec = '80C' -> section_limit('80C', Max), NewVal is C80C + Step, NewVal =< Max,
                     N80C = NewVal, N80D = C80D, NNPS = NPS
    ; Sec = '80D' -> section_limit('80D', Max), NewVal is C80D + Step, NewVal =< Max,
                     N80C = C80C, N80D = NewVal, NNPS = NPS
    ; Sec = '80CCD(1B)' -> section_limit('80CCD(1B)', Max), NewVal is NPS + Step, NewVal =< Max,
                           N80C = C80C, N80D = C80D, NNPS = NewVal ).

% Evaluate state tax
state_tax(state(C80C,C80D,NPS), Tax) :-
    Deductions = [('80C', C80C), ('80D', C80D), ('80CCD(1B)', NPS)],
    findall(_, retract(deduction(_,_)), _),
    forall(member((Sec,Val), Deductions), (Val > 0 -> assertz(deduction(Sec,Val)) ; true)),
    compute_tax(old, Tax),
    findall(_, retract(deduction(_,_)), _).

% Heuristic
heuristic(_State, H) :- total_income(Income), H is Income * 0.05.

% ---------- A* main ----------
astar_optimize(PlanCollapsed, BestTax) :-
    start_state(Start),
    empty_heap(Open0),
    state_tax(Start, StartTax),
    heuristic(Start, H0),
    F0 is StartTax + H0,
    add_to_heap(Open0, F0, node(Start, [], StartTax), Open),
    astar_loop(Open, [], RawPlan, BestTax),        % <-- collect raw plan
    collapse_plan(RawPlan, PlanCollapsed).         % <-- collapse here


% A* loop
astar_loop(Open, _Closed, Plan, BestTax) :-
    get_from_heap(Open, _F, node(State, Actions, G), _),
    \+ ( apply_action(State, _, Next), state_tax(Next, Tax), Tax < G ),
    reverse(Actions, Plan), BestTax = G, !.

astar_loop(Open, Closed, Plan, BestTax) :-
    get_from_heap(Open, _F, node(State, Actions, G), OpenRest),
    ( member(State, Closed) ->
        astar_loop(OpenRest, Closed, Plan, BestTax)
    ; findall(node(Next, [Act|Actions], NewG),
              ( apply_action(State, Act, Next), state_tax(Next, NewG) ),
              Children),
      add_children(Children, OpenRest, Open1),
      astar_loop(Open1, [State|Closed], Plan, BestTax) ).

% Add children
add_children([], Open, Open).
add_children([node(S,A,G)|Rest], Open0, Open) :-
    heuristic(S, H), F is G + H,
    add_to_heap(Open0, F, node(S,A,G), Open1),
    add_children(Rest, Open1, Open).

% ================================================================
% AO* Deduction Optimizer (fixed)
% ================================================================

% Entry point
ao_optimize(PlanCollapsed, BestTax) :-
    start_state(Start),
    ao_star(Start, Plan, BestTax, []),
    collapse_plan(Plan, PlanCollapsed).

% ---------- AO* ----------
% If state is terminal (no more actions), evaluate directly
ao_star(State, [], Tax, _) :-
    \+ apply_action(State, _, _),
    state_tax(State, Tax).

% Otherwise expand into ALL children (OR nodes)
ao_star(State, Plan, BestTax, Visited) :-
    findall((Act, SubPlan, SubTax),
            ( apply_action(State, Act, Next),
              \+ member(Next, Visited),
              ao_star(Next, SubPlan, SubTax, [State|Visited])
            ),
            Expansions),
    Expansions \= [],
    choose_best(Expansions, (BestAct, BestSubPlan, BestTax)),
    Plan = [BestAct|BestSubPlan].

% ---------- Choose best expansion ----------
choose_best([ (Act,SubPlan,SubTax) ], (Act,SubPlan,SubTax)).
choose_best([ (Act,SubPlan,SubTax) | Rest ], Best) :-
    choose_best(Rest, (A2,Plan2,Tax2)),
    ( SubTax < Tax2 ->
        Best = (Act,SubPlan,SubTax)
    ;   Best = (A2,Plan2,Tax2)
    ).

% ================================================================
% Unified Optimizer Wrapper
% ================================================================

% Call A* optimizer
optimize(astar, Plan, Tax) :-
    astar_optimize(Plan, Tax).

% Call AO* optimizer
optimize(ao, Plan, Tax) :-
    ao_optimize(Plan, Tax).


% ================================================================
% Bracket determination (fixed)
% ================================================================
bracket_from_ti(Regime, TI, Bracket) :-
    findall(Limit, tax_slab(Regime, Limit, _), Limits),
    sort(Limits, Sorted),
    bracket_from_limits(Regime, Sorted, TI, Bracket).

bracket_from_limits(Regime, [Limit|Rest], TI, Bracket) :-
    ( TI =< Limit -> slab_label(Regime, Limit, Bracket)
    ; bracket_from_limits(Regime, Rest, TI, Bracket) ).
bracket_from_limits(_, [], _, '30%').

% Old regime slab labels
slab_label(old, Limit, '0%')  :- Limit =< 250000, !.
slab_label(old, Limit, '5%')  :- Limit =< 500000, !.
slab_label(old, Limit, '20%') :- Limit =< 1000000, !.
slab_label(old, _, '30%').

% New regime slab labels
slab_label(new, Limit, '0%')  :- Limit =< 300000, !.
slab_label(new, Limit, '5%')  :- Limit =< 700000, !.
slab_label(new, Limit, '10%') :- Limit =< 1000000, !.
slab_label(new, Limit, '15%') :- Limit =< 1200000, !.
slab_label(new, Limit, '20%') :- Limit =< 1500000, !.
slab_label(new, _, '30%').

% ================================================================
% ML case export
% ================================================================
% case(Name, Age, Income, TotalDeductions, OldBracket, NewBracket)

save_current_case(Name) :-
    age(A), income(I), total_deductions(D),
    taxable_income(old, TI_old),
    taxable_income(new, TI_new),
    bracket_from_ti(old, TI_old, BOld),
    bracket_from_ti(new, TI_new, BNew),
    assertz(case(Name,A,I,D,BOld,BNew)).

% Export helper (write or append)
export_cases(File, Mode) :-
    ( Mode = write ->
        open(File, write, Stream),
        format(Stream, 'name,age,income,total_deductions,old_bracket,new_bracket~n', [])
    ; Mode = append ->
        open(File, append, Stream)
    ),
    forall(
        case(Name,A,I,D,BOld,BNew),
        format(Stream, '~w,~w,~w,~w,~w,~w~n', [Name,A,I,D,BOld,BNew])
    ),
    close(Stream).

% ================================================================
% Auto-generate synthetic training cases for ML
% ================================================================
generate_cases :-
    retractall(case(_,_,_,_,_,_)),
    forall(
        ( between(25, 70, BaseAge),
          member(BaseIncome, [200000,400000,600000,800000,1000000,1500000,2000000]),
          member(BaseD, [0,25000,50000,100000,150000])
        ),
        (
            random_between(-3, 3, NoiseAge),
            Age is max(18, BaseAge + NoiseAge),
            random_between(-25000, 25000, NoiseI),
            Income is BaseIncome + NoiseI,
            random_between(-5000, 5000, NoiseD),
            TempD is BaseD + NoiseD,
            (TempD < 0 -> D = 0 ; D = TempD),
            retractall(age(_)), retractall(income(_)), retractall(deduction(_,_)),
            assertz(age(Age)),
            assertz(income(Income)),
            assertz(deduction('80C', D)),
            ( catch(tax_summary, _, fail) ->
                gensym(case_, Name),
                save_current_case(Name)
            ; true )
        )
    ),
    % Always refresh synthetic dataset
    export_cases('synthetic_cases.csv', write),
    % Initialize cases.csv only if missing
    (   \+ exists_file('cases.csv')
    ->  export_cases('cases.csv', write)
    ;   true
    ).
