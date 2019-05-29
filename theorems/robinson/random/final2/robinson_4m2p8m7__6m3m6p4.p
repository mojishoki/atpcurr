%theorem: mul(plus(mul(4,2),8),7) = plus(mul(mul(6,3),6),4)
fof(zeroSuccessor,axiom, ! [X]: (o != s(X))).
fof(differentSuccessors,axiom, ! [X,Y]: (s(X) != s(Y) | X = Y)).
fof(additionZero,axiom, ! [X]: (plus(X,o) = X)).
fof(additionSuccessor,axiom, ! [X,Y]: (plus(X,s(Y)) = s(plus(X,Y)))).
fof(multiplicationZero,axiom, ! [X]: (mul(X,o) = o)).
fof(multiplicationSuccessor,axiom, ! [X,Y]: (mul(X,s(Y)) = plus(mul(X,Y),X))).
fof(myformula, conjecture, mul(plus(mul(s(s(s(s(o)))),s(s(o))),s(s(s(s(s(s(s(s(o))))))))),s(s(s(s(s(s(s(o)))))))) = plus(mul(mul(s(s(s(s(s(s(o)))))),s(s(s(o)))),s(s(s(s(s(s(o))))))),s(s(s(s(o)))))).