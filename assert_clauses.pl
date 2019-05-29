%%% write clauses into Prolog's database
assert_clauses([],_).
assert_clauses([C|M],Set) :-
    (Set\=conj, \+member(-_,C) -> C1=[#|C] ; C1=C),
    (ground(C) -> G=g ; G=n), assert_clauses2(C1,[],G),
    assert_clauses(M,Set).

assert_clauses2([],_,_).
assert_clauses2([L|C],C1,G) :-
    assert_renvar([L],[L2]), append(C1,C,C2), append(C1,[L],C3),
    assert(lit(L2,L,C2,G)), assert_clauses2(C,C3,G).

assert_renvar([],[]).
assert_renvar([F|FunL],[F1|FunL1]) :-
    ( var(F) -> true ; F=..[Fu|Arg], assert_renvar(Arg,Arg1),
      F1=..[Fu|Arg1] ), assert_renvar(FunL,FunL1).
