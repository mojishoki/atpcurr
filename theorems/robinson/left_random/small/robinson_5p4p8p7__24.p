%theorem: plus(plus(plus(5,4),8),7) = 24
fof(zeroSuccessor,axiom, ! [X]: (o != s(X))).
fof(differentSuccessors,axiom, ! [X,Y]: (s(X) != s(Y) | X = Y)).
fof(additionZero,axiom, ! [X]: (plus(X,o) = X)).
fof(additionSuccessor,axiom, ! [X,Y]: (plus(X,s(Y)) = s(plus(X,Y)))).
fof(multiplicationZero,axiom, ! [X]: (mul(X,o) = o)).
fof(multiplicationSuccessor,axiom, ! [X,Y]: (mul(X,s(Y)) = plus(mul(X,Y),X))).
cnf(myformula, negated_conjecture, plus(plus(plus(s(s(s(s(s(o))))),s(s(s(s(o))))),s(s(s(s(s(s(s(s(o))))))))),s(s(s(s(s(s(s(o)))))))) != s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(s(o))))))))))))))))))))))))).