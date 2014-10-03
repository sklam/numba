from __future__ import print_function, division, absolute_import

from numba import utils
from numba.bytecode import ByteCodeInst, CustomByteCode


def lift_loop(bytecode, dispatcher_factory):
    """Lift the top-level loops.

    Returns (outer, loops)
    ------------------------
    * outer: ByteCode of a copy of the loop-less function.
    * loops: a list of ByteCode of the loops.
    """
    outer = []
    loops = []
    separate_loops(bytecode, outer, loops)

    # Discover variables references
    outer_rds, outer_wrs = find_varnames_uses(bytecode, outer)
    outer_wrs |= set(bytecode.argspec.args)

    dispatchers = []
    outerlabels = set(bytecode.labels)
    outernames = list(bytecode.co_names)

    for loop in loops:
        args, rets = discover_args_and_returns(bytecode, loop, outer_rds,
                                               outer_wrs)

        disp = insert_loop_call(bytecode, loop, args,
                                outer, outerlabels, rets,
                                dispatcher_factory)
        dispatchers.append(disp)

    # Build outer bytecode
    codetable = utils.SortedMap((i.offset, i) for i in outer)
    outerbc = CustomByteCode(func=bytecode.func,
                             func_qualname=bytecode.func_qualname,
                             argspec=bytecode.argspec,
                             filename=bytecode.filename,
                             co_names=outernames,
                             co_varnames=bytecode.co_varnames,
                             co_consts=bytecode.co_consts,
                             co_freevars=bytecode.co_freevars,
                             table=codetable,
                             labels=outerlabels & set(codetable.keys()))

    print(outerbc.dump())
    return outerbc, dispatchers

@utils.total_ordering
class SubOffset(object):
    def __init__(self, val, sub=1):
        assert sub > 0
        self.val = val
        self.sub = sub

    def next(self):
        return SubOffset(self.val, self.sub + 1)

    def __add__(self, other):
        return SubOffset(self.val, self.sub + other)

    def __hash__(self):
        return hash((self.val, self.sub))

    def __lt__(self, other):
        if isinstance(other, SubOffset):
            if self.val < other.val:
                return self
            elif self.val == other.val:
                return self.sub < other.sub
            else:
                return False
        else:
            return self.val < other

    def __eq__(self, other):
        if isinstance(other, SubOffset):
            return self.val == other.val and self.sub == other.sub
        else:
            return False

    def __repr__(self):
        return "{0}.{1}".format(self.val, self.sub)


def insert_loop_call(bytecode, loop, args, outer, outerlabels, returns,
                     dispatcher_factory):
    endloopoffset = loop[-1].next
    # Accepted. Create a bytecode object for the loop
    args = tuple(args)

    lbc = make_loop_bytecode(bytecode, loop, args, returns)

    # Generate dispatcher for this inner loop, and append it to the
    # consts tuple.
    disp = dispatcher_factory(lbc)
    disp_idx = len(bytecode.co_consts)
    bytecode.co_consts += (disp,)

    # Insert jump to the end
    insertpt = SubOffset(loop[0].next)
    jmp = ByteCodeInst.get(loop[0].offset, 'JUMP_ABSOLUTE', insertpt)
    jmp.lineno = loop[0].lineno
    insert_instruction(outer, jmp)

    outerlabels.add(outer[-1].next)

    # Prepare arguments
    loadfn = ByteCodeInst.get(insertpt, "LOAD_CONST", disp_idx)
    loadfn.lineno = loop[0].lineno
    insert_instruction(outer, loadfn)

    insertpt = insertpt.next()
    for arg in args:
        loadarg = ByteCodeInst.get(insertpt, 'LOAD_FAST',
                                   bytecode.co_varnames.index(arg))
        loadarg.lineno = loop[0].lineno
        insert_instruction(outer, loadarg)
        insertpt = insertpt.next()

    # Call function
    assert len(args) < 256
    call = ByteCodeInst.get(insertpt, "CALL_FUNCTION", len(args))
    call.lineno = loop[0].lineno
    insert_instruction(outer, call)

    insertpt = insertpt.next()

    if returns:
        # Unpack arguments
        unpackseq = ByteCodeInst.get(insertpt, "UNPACK_SEQUENCE",
                                  len(returns))
        unpackseq.lineno = loop[0].lineno
        insert_instruction(outer, unpackseq)
        insertpt = insertpt.next()

        for out in returns:
            # Store each variable
            storefast = ByteCodeInst.get(insertpt, "STORE_FAST",
                                      bytecode.co_varnames.index(out))
            storefast.lineno = loop[0].lineno
            insert_instruction(outer, storefast)
            insertpt = insertpt.next()
    else:
        # No return value
        poptop = ByteCodeInst.get(outer[-1].next, "POP_TOP", None)
        poptop.lineno = loop[0].lineno
        insert_instruction(outer, poptop)
        insertpt = insertpt.next()

    jmpback = ByteCodeInst.get(insertpt, 'JUMP_ABSOLUTE',
                               endloopoffset)

    jmpback.lineno = loop[0].lineno
    insert_instruction(outer, jmpback)

    return disp


def insert_instruction(insts, item):
    i = find_previous_inst(insts, item.offset)
    insts.insert(i, item)


def find_previous_inst(insts, offset):
    for i, inst in enumerate(insts):
        if inst.offset > offset:
            return i
    return len(insts)


def make_loop_bytecode(bytecode, loop, args, returns):
    # Add return None
    co_consts = tuple(bytecode.co_consts)
    if None not in co_consts:
        co_consts += (None,)

    if returns:
        for out in returns:
            # Load output
            loadfast = ByteCodeInst.get(loop[-1].next, "LOAD_FAST",
                                         bytecode.co_varnames.index(out))
            loadfast.lineno = loop[-1].lineno
            loop.append(loadfast)
            # Build tuple
            buildtuple = ByteCodeInst.get(loop[-1].next, "BUILD_TUPLE",
                                        len(returns))
            buildtuple.lineno = loop[-1].lineno
            loop.append(buildtuple)

    else:
        # Load None
        load_none = ByteCodeInst.get(loop[-1].next, "LOAD_CONST",
                                     co_consts.index(None))
        load_none.lineno = loop[-1].lineno
        loop.append(load_none)

    # Return TOS
    return_value = ByteCodeInst.get(loop[-1].next, "RETURN_VALUE", 0)
    return_value.lineno = loop[-1].lineno
    loop.append(return_value)

    # Function name
    loop_qualname = bytecode.func_qualname + ".__numba__loop%d__" % loop[0].offset

    # Argspec
    argspectype = type(bytecode.argspec)
    argspec = argspectype(args=args, varargs=(), keywords=(), defaults=())

    # Code table
    codetable = utils.SortedMap((i.offset, i) for i in loop)

    # Custom bytecode object
    lbc = CustomByteCode(func=bytecode.func,
                         func_qualname=loop_qualname,
                         argspec=argspec,
                         filename=bytecode.filename,
                         co_names=bytecode.co_names,
                         co_varnames=bytecode.co_varnames,
                         co_consts=co_consts,
                         co_freevars=bytecode.co_freevars,
                         table=codetable,
                         labels=bytecode.labels)

    return lbc


def stitch_instructions(outer, loop):
    begin = loop[0].offset
    i = find_previous_inst(outer, begin)
    return outer[:i] + loop + outer[i:]


def discover_args_and_returns(bytecode, insts, outer_rds, outer_wrs):
    """
    Basic analysis for args and returns
    This completely ignores the ordering or the read-writes.
    """
    rdnames, wrnames = find_varnames_uses(bytecode, insts)
    # Pass names that are written outside and read locally
    args = outer_wrs & rdnames
    # Return values that it written locally and read outside
    rets = wrnames & outer_rds
    return args, rets


def find_varnames_uses(bytecode, insts):
    rdnames = set()
    wrnames = set()
    for inst in insts:
        if inst.opname == 'LOAD_FAST':
            rdnames.add(bytecode.co_varnames[inst.arg])
        elif inst.opname == 'STORE_FAST':
            wrnames.add(bytecode.co_varnames[inst.arg])
    return rdnames, wrnames


def separate_loops(bytecode, outer, loops):
    """
    Separate top-level loops from the function

    Stores loopless instructions from the original function into `outer`.
    Stores list of loop instructions into `loops`.
    Both `outer` and `loops` are list-like (`append(item)` defined).
    """
    endloop = None
    cur = None
    for inst in bytecode:
        if endloop is None:
            if inst.opname == 'SETUP_LOOP':
                cur = [inst]
                # Python may set the end of loop to the final jump destination
                # when nested in a if-else.  We need to scan the bytecode to
                # find the actual end of loop
                endloop = _scan_real_end_loop(bytecode, inst)
            else:
                outer.append(inst)
        else:
            cur.append(inst)
            if inst.next == endloop:
                for inst in cur:
                    if inst.opname == 'RETURN_VALUE':
                        # Reject if return inside loop
                        outer.extend(cur)
                        break
                else:
                    loops.append(cur)
                endloop = None


def _scan_real_end_loop(bytecode, setuploop_inst):
    """Find the end of loop.
    Return the instruction offset.
    """
    start = setuploop_inst.next
    end = start + setuploop_inst.arg
    offset = start
    depth = 0
    while offset < end:
        inst = bytecode[offset]
        depth += inst.block_effect
        if depth < 0:
            return inst.next
        offset = inst.next

