import argparse
import os
import math
import numpy as np
# np.random.seed(100)

def process_op_list(ops):
    if "," in ops: 
        ops = ops.split(",")
    else:
        ops = [ops, ]
    ops = [o.split("|") for o in ops]
    op_list = []
    op_limits = {}
    for (op, limit) in ops:
        op_list.append(op)
        op_limits[op] = int(limit)
    return tuple(op_list), op_limits
    
def generate_operators(ops, op_count, useall=False):
    if useall:
        curr_ops = np.random.permutation(ops)
    else:
        indices = np.random.randint(low=0, high=len(ops), size=op_count)
        curr_ops = np.array(ops)[indices]
    return tuple(curr_ops)

def generate_numbers(curr_ops, first_limit, op_limits):
    curr_value = np.random.randint(low=0, high=first_limit)
    numbers = [curr_value]
    for op in curr_ops:
        limit = op_limits[op]
        if op == 'plus':
            n = np.random.randint(low=0, high=limit)
            numbers.append(n)
            curr_value += n
        elif op == 'mul':
            n = np.random.randint(low=0, high=limit)
            numbers.append(n)
            curr_value *= n
        elif op == 'exp':
            n = np.random.randint(low=0, high=limit)
            numbers.append(n)
            curr_value **= n
        elif op == 'neg':
            n = np.random.randint(low=0, high=min(limit,curr_value+1))
            numbers.append(n)
            curr_value -= n
        elif op == 'div':
            while True:
                n = np.random.randint(low=1, high=limit)
                if curr_value % n == 0:
                    break
            numbers.append(n)
            curr_value //= n
        elif op == 'log':
            assert False, "Not Yet implemented"
        else:
            assert False, "Unknown operator: " + op
    return tuple(numbers), curr_value

def decimal_to_unary_string(decimal):
    assert decimal >= 0
    prefix = ""
    suffix = ""
    for i in range(decimal):
        prefix += "s("
        suffix += ")"
    return prefix + "o" + suffix
    # if decimal == 0:
    #     return 'o'
    # else:
    #     return "s({})".format(decimal_to_unary_string(decimal-1))

def turn_to_formula(numbers, ops):
    assert len(numbers) == len(ops) + 1
    formula = decimal_to_unary_string(numbers[0])
    formula_readable = str(numbers[0])
    for i, op in enumerate(ops):
        formula = "{}({},{})".format(op, formula, decimal_to_unary_string(numbers[i+1]))
        formula_readable = "{}({},{})".format(op, formula_readable, str(numbers[i+1]))
    return formula, formula_readable
    
def build_fof(numbers, ops, value):
    left_side, left_side_readable = turn_to_formula(numbers, ops)
    right_side = decimal_to_unary_string(value)
    right_side_readable = str(value)
    conjecture = "{} = {}".format(left_side, right_side)
    fof = "fof(myformula, conjecture, {}).".format(conjecture)
    comment = "%theorem: {} = {}\n".format(left_side_readable, right_side_readable)
    return fof, comment

def build_pair_fof(numbers1, ops1, numbers2, ops2):
    left_side, left_side_readable = turn_to_formula(numbers1, ops1)
    right_side, right_side_readable = turn_to_formula(numbers2, ops2)
    conjecture = "{} = {}".format(left_side, right_side)
    fof = "fof(myformula, conjecture, {}).".format(conjecture)
    comment = "%theorem: {} = {}\n".format(left_side_readable, right_side_readable)
    return fof, comment

def build_name_term(numbers, ops):
    assert len(numbers) == len(ops) + 1
    result = str(numbers[0])
    for i, op in enumerate(ops):
        result += op[0]
        result += str(numbers[i+1])
    return result

def build_name(numbers, ops, value):
    result = build_name_term(numbers, ops)
    result += "__" + str(value)
    return result

def build_name_pair(numbers1, ops1, numbers2, ops2):
    result1 = build_name_term(numbers1, ops1)
    result2 = build_name_term(numbers2, ops2)
    result = result1 + "__" + result2
    return result

def generate_terms(term_count, op_list, op_count, useall, first_limit, op_limits):
    term_map = {}
    for _ in range(term_count):
        curr_ops = generate_operators(op_list, op_count, useall)
        curr_numbers, curr_value = generate_numbers(curr_ops, first_limit, op_limits)
        term_map.setdefault(curr_value, [])
        item = (curr_numbers, curr_ops)
        if not item in term_map[curr_value]:
            term_map[curr_value].append(item)
    return term_map

# builds f=N where f is a random formula and N is an number
def build_random(args, prefix, preamble):
    op_list, op_limits = process_op_list(args.ops)
    for i in range(args.count):
        curr_ops = generate_operators(op_list, args.op_count, args.useall)
        curr_numbers, curr_value = generate_numbers(curr_ops, args.first_limit, op_limits)
        formula, comment = build_fof(curr_numbers, curr_ops, curr_value)
        name = build_name(curr_numbers, curr_ops, curr_value)
        print("\n", comment, formula, "\n", name)
        write_to_file(args.output_dir, preamble, prefix, comment, formula, name)

# builds f1=f2 where both f1 and f2 are random formulae that have the same value
def build_pairs(args, prefix, preamble):
    op_list, op_limits = process_op_list(args.ops)
    term_map = generate_terms(args.count, op_list, args.op_count, args.useall, args.first_limit, op_limits)
    for value in term_map.keys():
        terms = term_map[value]
        leftIndex = 0
        rightIndex = 1
        while True:
            if leftIndex >= len(terms):
                break
            elif rightIndex >= len(terms):
                leftIndex += 1
                rightIndex = leftIndex + 1
            else:
                numbers1, ops1 = terms[leftIndex]
                numbers2, ops2 = terms[rightIndex]
                formula, comment = build_pair_fof(numbers1, ops1, numbers2, ops2)
                rightIndex += 1
                name = build_name_pair(numbers1, ops1, numbers2, ops2)
                print("\n", comment, formula, "\n", name)
                write_to_file(args.output_dir, preamble, prefix, comment, formula, name)

def write_to_file(output_dir, preamble, prefix, comment, formula, name):
    f_out = open(os.path.join(output_dir,"{}_{}.p".format(prefix, name)), "w")
    f_out.write(comment)
    f_out.write(preamble)
    f_out.write(formula)
    f_out.close()

parser = argparse.ArgumentParser()
parser.add_argument("--count", dest="count", type=int, default=10, help="Number of formulae to generate")
parser.add_argument("--first_limit", dest="first_limit", type=int, default=10, help="Limit of the first number of the formula")
parser.add_argument("--op_count", dest="op_count", type=int, default=1, help="The number of operators used in the formula")
parser.add_argument("--ops", dest="ops", default="plus|5,neg|5,mul|5,div|5,exp|2", help="Comma separated list of available operators|limit pairs where limit refers to the right argument (plus, neg, mul, div, exp, log)")
parser.add_argument("--preamble_file", dest="preamble_file", default="peano_extended.p", help="The preamble file")
parser.add_argument("--output_dir", dest="output_dir", default="../theorems/arithmetic", help="The output directory")
parser.add_argument("--useall", dest="useall", default=0, type=int, help="If > 0 then use each operator exactly once. Useful for generating learning problems")
parser.add_argument("--type", dest="type", default="onesided", help="onesides/pairs: term=N / term1=term2")

args = parser.parse_args()

for a in vars(args):
    print("{}: {}".format(a, vars(args)[a]))
print("\n\n")

f_in = open(args.preamble_file, "r") 
preamble = f_in.read()

if args.preamble_file == "peano_fof.p":
    prefix = "robinson"
elif args.preamble_file == "peano_extended_fof.p":
    prefix = "extended"
else:
    assert False


if args.type == "onesided":
    result = build_random(args, prefix, preamble)
elif args.type == "pairs":
    result = build_pairs(args, prefix, preamble)
else:
    assert False


