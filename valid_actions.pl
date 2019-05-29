% valid_actions(+Goal,+Path,-Actions)
valid_actions2([],_,[]).
valid_actions2([Lit|_],Path,Actions):-
    (-NegLit=Lit;-Lit=NegLit), !,
    valid_actions(NegLit,Path,Actions).

% valid_actions(+NegLit,+Path,-Actions)
% Actions is all non-deterministic (reduction and extension) steps
valid_actions(NegLit,Path,Actions):-
    select_unifiable(Path,NegLit,Reductions),
    select_unifiable_contras(NegLit,Extensions),
    append(Reductions,Extensions,Actions).


% select_unifiable_reds(List, E, Res):
% Res is a subsequence of List wrapped in red/1 such that elements of Res are all unifiable with E
select_unifiable([],_,[]).
select_unifiable([L|List], E, Res):-
    ( \+(unify_with_occurs_check(L, E)) -> Res=Res2
     ;Res=[red(L)|Res2]
    ),
    select_unifiable(List,E,Res2).

select_unifiable_contras(E, Contras):-
    findall( ext(NegL,Cla1,Grnd1), (
                 lit(E,NegL,Cla1,Grnd1),
                 \+(\+(unify_with_occurs_check(E,NegL)))
             ), Contras
           ).
