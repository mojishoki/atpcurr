fof(additionZero, axiom, ! [X]: plus(X, o) = X).
fof(additionSuccessor, axiom, ! [X, Y]: plus(X, s(Y)) = s(plus(X, Y))).
fof(multiplicationZero, axiom, ! [X]: mul(X, o) = o).
fof(multiplicationSuccessor, axiom, ! [X, Y]: mul(X, s(Y)) = plus(mul(X, Y), X)).