:- dynamic(lit/4).
:- dynamic(option/1).
:- dynamic(state/6).

:- [features]. % enigma features
:- [leancop_tptp2].  % load program for TPTP input syntax
:- [def_mm].  % load program for clausal form translation
:- [assert_clauses]. % load program for asserting clauses
:- [valid_actions].

embed_init:-
    Settings = [conj,nodef,verbose,print_proof,n_dim(100)],
    embed_init('theorems/peano1plus1.p', Settings).
embed_init(File,Settings):-
    embed_init(File,Settings,_State,_Actions,_Result).
embed_init(File,Settings,[EGoal,EPath,ELem,ETodos],EActions,Result):-
    init(File,Settings,[Goal,Path,Lem,Todos],Actions,Result),
    copy_term([Goal,Path,Lem,Todos,Actions],[Goal1,Path1,Lem1,Todos1,Actions1]),
    numbervars([Goal1,Path1,Lem1,Todos1,Actions1],1000,_),
    option(n_dim(FDim)),
    cached_embed2(Goal1, FDim,0,EGoal),
    cached_embed2(Path1, FDim,0,EPath),
    cached_embed2(Lem1,  FDim,0,ELem),
    cached_embed2(Todos1,FDim,0,ETodos),
    cached_embed_list2(Actions1,FDim,0,EActions).
embed_step(ActionIndex,[EGoal,EPath,ELem,ETodos],EActions,Result):-
    step(ActionIndex,[Goal,Path,Lem,Todos],Actions,Result),
    copy_term([Goal,Path,Lem,Todos,Actions],[Goal1,Path1,Lem1,Todos1,Actions1]),
    numbervars([Goal1,Path1,Lem1,Todos1,Actions1],1000,_),
    option(n_dim(FDim)),
    cached_embed2(Goal1, FDim,0,EGoal),
    cached_embed2(Path1, FDim,0,EPath),
    cached_embed2(Lem1,  FDim,0,ELem),
    cached_embed2(Todos1,FDim,0,ETodos),
    cached_embed_list2(Actions1,FDim,0,EActions).

init:-
    Settings = [conj,nodef,verbose,print_proof,n_dim(10)],
    init('theorems/peano1plus1.p', Settings).
init(File,Settings):-
    init(File,Settings,_,_,_).
init(File,Settings,[Goal,Path,Lem,Todos],Actions,Result):-
    init_pure(File,Settings,NewState),
    NewState = state(Goal,Path,Lem,Actions,Todos,Proof,Result),
    set_state(Goal,Path,Lem,Actions,Todos,Proof),
    log(Goal,Path,Lem,Actions,Todos,Proof,Result,start).

% init_pure(+File,+Settings,-NewState)
init_pure(File,Settings,NewState):-
    NewState = state(Goal,Path,Lem,Actions,Todos,Proof,Result),

    retractall(option(_)),
    findall(_, ( member(S,Settings), assert(option(S)) ), _ ),

    AxPath='', AxNames=[_],
    leancop_tptp2(File,AxPath,AxNames,Problem,Conj), !,
    ( Conj\=[] -> Problem1=Problem ; Problem1=(~Problem) ),
    leancop_equal(Problem1,Problem2),
    make_matrix(Problem2,Matrix,Settings),
    ( option(verbose) ->
	  writeln(["Problem ", Problem2]),
	  writeln(["Matrix ", Matrix])
     ; true
    ),
    retractall(lit(_,_,_,_)),
    (member([-(#)],Matrix) -> S=conj ; S=pos),
    assert_clauses(Matrix,S),
    det_steps([-(#)],[],[],[],[],Goal,Path,Lem,Todos,Proof,Result0),
    valid_actions2(Goal,Path,Actions),
    (  length(Actions,0), Result0 < 1 -> Result = -1
     ; Result = Result0
    ).


:- dynamic(alternative/6).
step(ActionIndex):-
    step(ActionIndex,_,_,_).
step(ActionIndex,[Goal,Path,Lem,Todos],Actions,Result):-
    state(Goal0,Path0,Lem0,Actions0,Todos0,Proof0),
    State = state(Goal0,Path0,Lem0,Actions0,Todos0,Proof0,_Result0),
    step_pure(ActionIndex,State,NewState,Action0),
    NewState = state(Goal,Path,Lem,Actions,Todos,Proof,Result),
					 
    set_state(Goal,Path,Lem,Actions,Todos,Proof),
    log(Goal,Path,Lem,Actions,Todos,Proof,Result,Action0).


% step_pure(+ActionIndex,+State,-NewState,-SelectedAction))
step_pure(ActionIndex,State,NewState,Action0):-
    ( State = state(Goal0,Path0,Lem0,Actions0,Todos0,Proof0,_Result0) ->
      NewState = state(Goal,Path,Lem,Actions,Todos,Proof,Result)
    ; State = state(Goal0,Path0,Lem0,Actions0,Todos0,_Result0) ->
      NewState = state(Goal,Path,Lem,Actions,Todos,Result)
    ),

    nth0(ActionIndex,Actions0,Action0),

    % if there were other alternative actions, store them as alternatives
    (option(backtrack), Actions0=[_,_|_] ->
	 select_nounif(Action0, Actions0, RemActions0), !,
	 asserta(alternative(Goal0,Path0,Lem0,RemActions0,Todos0,Proof0))
     ; true
    ),
    
    nondet_step(Action0,Goal0,Path0,Lem0,Todos0,Proof0,Goal1,Path1,Lem1,Todos1,Proof1,Result1),
    ( Result1 == -1, option(backtrack), pop_alternative(Goal,Path,Lem,Actions,Todos,Proof) ->
	  Result=0,
	  log(Goal,Path,Lem,Actions,Todos,Proof,Result,Action0)
												   
     ; [Goal,Path,Lem,Todos,Proof,Result] = [Goal1,Path1,Lem1,Todos1,Proof1,Result1],
       valid_actions2(Goal,Path,Actions)
    ).


pop_alternative(Goal,Path,Lem,Actions,Todos,Proof):-
    alternative(Goal,Path,Lem,Actions,Todos,Proof),
    retract(alternative(Goal,Path,Lem,Actions,Todos,Proof)), !.

:- dynamic(state/6).
% save the current state
set_state(Goal,Path,Lem,Actions,Todos,Proof):-
    retractall(state(_,_,_,_,_,_)),
    assert(state(Goal,Path,Lem,Actions,Todos,Proof)).
% log exploration
log(Goal,Path,Lem,Actions,Todos,Proof,Result,Selected):-
    ( option(verbose) ->
	  writeln(["Selected ", Selected]),
	  writeln(["State ", Goal, Path, Lem]),
      foreach(member(A,Actions), format("Action ~w\n",[A])),
	  % writeln(["Actions ", Actions]),
	  writeln(["Result ", Result]),
	  writeln(["Todos ", Todos])
     ; true
    ),

    ( option(print_proof), Result == 1 ->
	  writeln("Proof found:"),
	  reverse(Proof,ProofRev),
	  print_proof(ProofRev)
     ; true
    ).


% goal and path share an identical literal
has_loop(Goal,Path):-
    (member(LitC,Goal), member(LitP,Path), LitC==LitP), !.
% Lit is the negation of NegLit
neg_lit(Lit, NegLit):-
    (-NegLit=Lit;-Lit=NegLit), !.
select_nounif(E,Xs,Rem):-
    select_nounif(Xs,E,[],Rem).
select_nounif([X|Xs],E,Acc,Rem):-
    ( X == E -> reverse(Acc,RevAcc), append(RevAcc,Xs,Rem)
     ;select_nounif(Xs,E,[X|Acc],Rem)
    ).



%%% make a single proof step from a choice point
% nondet_step(Action,Goal,Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result)
nondet_step(red(NegL), [Lit|Cla],Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result):- % reduction step
    copy_term(NegL,NegL_orig),
    neg_lit(Lit,NegL),
    Proof2 = [red(NegL_orig-NegL)|Proof],
    det_steps(Cla,Path,Lem,Todos,Proof2,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result).
nondet_step(ext(NegLit,Cla1,_Grnd1), [Lit|Cla],Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result):- % extension step
    copy_term([NegLit|Cla1], Ext_orig),
    neg_lit(Lit, NegLit),
    ( Cla=[_|_] ->
	  Todos2 = [[Cla,Path,[Lit|Lem]]|Todos]
     ; Todos2 = Todos
    ),
    Proof2=[ext(Ext_orig-[NegLit|Cla1])|Proof],
    det_steps(Cla1,[Lit|Path],Lem,Todos2,Proof2,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result).

% perform steps until the next choice point (or end of proof)
det_steps([],_Path,_Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result):-
    !,
    ( Todos = [] -> % nothing to prove, nothing todo on the stack
	  [NewGoal,NewPath,NewLem,NewTodos,NewProof,Result] = [[success],[],[],[],Proof,1]
     ; Todos = [[Goal2,Path2,Lem2]|Todos2] -> % nothing to prove, something on the stack
	   det_steps(Goal2,Path2,Lem2,Todos2,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result)
    ).
det_steps([Lit|Cla],Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result):-
    member(LitL,Lem), Lit==LitL, !, % perform lemma step
    Proof2 = [lem(Lit)|Proof],
    det_steps(Cla,Path,Lem,Todos,Proof2,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result).
det_steps([Lit|Cla],Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result):-
    neg_lit(Lit,NegLit),
    member(NegL,Path), NegL == NegLit, !, % reduction step without unification can be performed eagerly
    Proof2 = [red(NegL-NegL)|Proof],
    det_steps(Cla,Path,Lem,Todos,Proof2,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result).
det_steps(Goal,Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result):-
    valid_actions2(Goal,Path,Actions),
    ( option(single_action_optim),  Actions==[A] -> % only a single action is available, so perform it
	  nondet_step(A,Goal,Path,Lem,Todos,Proof,NewGoal,NewPath,NewLem,NewTodos,NewProof,Result)
     ;Actions==[] -> % proof failed
	  [NewGoal,NewPath,NewLem,NewTodos,NewProof,Result] = [[failure],[],[],[],[],-1]
     ;[NewGoal,NewPath,NewLem,NewTodos,NewProof,Result] = [Goal,Path,Lem,Todos,Proof,0]
    ).
    

% embed formulas into sequences of ints
% THIS IS A VERY SIMPLE, VERY PRELIMINARY VERSION
% variables are marked with 0
% other symbols are marked with a unique index
:- dynamic(atomlist/1).

embed(X,Emb):-
    ( atomlist(AtomL0) -> true
     ;AtomL0=[]
    ),
    embed(X,AtomL0,[],AtomL,Emb),
    retractall(atomlist(_)),
    assert(atomlist(AtomL)).
embed(X,AtomL0,Emb0,AtomL,Emb):-
    ( var(X) -> Name = var
     ;atom(X) -> Name = X
    ), !,
    add_to_embedding(Name,AtomL0,Emb0,AtomL,Emb).
embed(X,AtomL0,Emb0,AtomL,Emb):-
    is_list(X), !,
    embed_list(X,AtomL0,Emb0,AtomL,Emb).
embed(X,AtomL0,Emb0,AtomL,Emb):-
    X=..[Name|Args],
    add_to_embedding(Name,AtomL0,Emb0,AtomL1,Emb1),
    embed_list(Args,AtomL1,Emb1,AtomL,Emb).

embed_list([],AtomL,Emb,AtomL,Emb).
embed_list([X|Xs],AtomL0,Emb0,AtomL,Emb):-
    embed(X,AtomL0,Emb0,AtomL1,Emb1),
    embed_list(Xs,AtomL1,Emb1,AtomL,Emb).
		 
add_to_embedding(Name,AtomL0,Emb0,AtomL,Emb):-
    ( nth0(I,AtomL0,Name) -> AtomL=AtomL0
     ;append(AtomL0,[Name],AtomL),
      nth0(I,AtomL,Name)
    ),
    Emb=[I|Emb0].

% print proofs
print_proof([]).
print_proof([init(Orig-Substituted)|Proof]):- !,
    format('   ~w: ~t ~w -> ~w\n', ['Init     ', Orig, Substituted]),
    print_proof(Proof).
print_proof([lem(Lit)|Proof]):- !,
    format('   ~w: ~t ~w\n', ['Lemma    ', Lit]),
    print_proof(Proof).
print_proof([red(Orig-Substituted)|Proof]):- !,
    format('   ~w: ~t ~w -> ~w\n', ['Reduction', Orig, Substituted]),
    print_proof(Proof).
print_proof([ext(Orig-Substituted)|Proof]):- !,
    format('   ~w: ~t ~w -> ~w\n', ['Extension', Orig, Substituted]),
    print_proof(Proof).
