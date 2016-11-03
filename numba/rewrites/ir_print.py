from __future__ import print_function

from numba import ir, errors
from . import register_rewrite, InstRewrite


@register_rewrite('before-inference')
class RewritePrintCalls(InstRewrite):
    """
    Rewrite calls to the print() global function to dedicated IR print() nodes.
    """
    rewrite_inst_of = ir.Assign

    def match_inst(self, interp, inst, typemap, calltypes):
        # Find all assignments with a right-hand print() call
        if isinstance(inst.value, ir.Expr):
            expr = inst.value
            if expr.op == 'call':
                if expr.kws:
                    # Only positional args are supported
                    return
                try:
                    callee = interp.infer_constant(expr.func)
                except errors.ConstantInferenceError:
                    return
                if callee is print:
                    return expr

    def rewrite_inst(self, inst, expr):
        """
        Rewrite `var = call <print function>(...)` as a sequence of
        `print(...)` and `var = const(None)`.
        """
        yield ir.Print(args=expr.args, vararg=expr.vararg, loc=expr.loc)
        yield ir.Assign(value=ir.Const(None, loc=expr.loc), target=inst.target,
                        loc=inst.loc)


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
