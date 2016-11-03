from __future__ import print_function

from numba import ir, errors
from . import register_rewrite, Rewrite, InstRewrite


@register_rewrite('before-inference')
class RewritePrintCalls(Rewrite):
    """
    Rewrite calls to the print() global function to dedicated IR print() nodes.
    """

    def match(self, interp, block, typemap, calltypes):
        self.prints = prints = {}
        self.block = block
        # Find all assignments with a right-hand print() call

        for m, inst, expr in block.match_exprs(op='call'):
            if m:
                expr = inst.value
                if expr.kws:
                    # Only positional args are supported
                    continue
                try:
                    callee = interp.infer_constant(expr.func)
                except errors.ConstantInferenceError:
                    continue
                if callee is print:
                    prints[inst] = expr
        return len(prints) > 0

    def apply(self, new_block):
        """
        Rewrite `var = call <print function>(...)` as a sequence of
        `print(...)` and `var = const(None)`.
        """
        for m, inst, expr in self.block.match_exprs(op='call'):
            if m and inst in self.prints:
                print_node = ir.Print(args=expr.args, vararg=expr.vararg,
                                      loc=expr.loc)
                new_block.append(print_node)
                assign_node = ir.Assign(value=ir.Const(None, loc=expr.loc),
                                        target=inst.target,
                                        loc=inst.loc)
                new_block.append(assign_node)
            else:
                new_block.append(inst)


@register_rewrite('before-inference')
class DetectConstPrintArguments(InstRewrite):
    """
    Detect and store constant arguments to print() nodes.
    """
    rewrite_inst_of = ir.Print

    def match_inst(self, interp, inst, typemap, calltypes):
        if inst.consts:
            # Already rewritten
            return
        out = {}
        for idx, var in enumerate(inst.args):
            try:
                const = interp.infer_constant(var)
            except errors.ConstantInferenceError:
                continue
            out[idx] = const
        if out:
            return out

    def rewrite_inst(self, inst, consts):
        """
        Store detected constant arguments on their nodes.
        """
        new_inst = inst.copy()
        new_inst.consts = consts
        return new_inst
