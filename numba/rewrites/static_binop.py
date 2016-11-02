from numba import ir, errors
from . import register_rewrite, Rewrite


@register_rewrite('before-inference')
class DetectStaticBinops(Rewrite):
    """
    Detect constant arguments to select binops.
    """

    # Those operators can benefit from a constant-inferred argument
    rhs_operators = {'**'}

    def match(self, interp, block, typemap, calltypes):
        self.static_lhs = {}
        self.static_rhs = {}
        self.block = block
        # Find binop expressions with a constant lhs or rhs
        for expr in block.find_exprs(op='binop'):
            try:
                if (expr.fn in self.rhs_operators
                    and expr.static_rhs is ir.UNDEFINED):
                    self.static_rhs[expr] = interp.infer_constant(expr.rhs)
            except errors.ConstantInferenceError:
                continue

        return len(self.static_lhs) > 0 or len(self.static_rhs) > 0

    def apply(self):
        """
        Store constant arguments that were detected in match().
        """
        new_block = self.block.copy()
        new_block.clear()
        for m, inst, expr in self.block.match_exprs(op='binop'):
            new_inst = inst.copy()
            if m and expr in self.static_rhs:
                new_expr = expr.copy()
                new_expr.static_rhs = expr.rhs
                new_inst.value = new_expr
            new_block.append(new_inst)
        return new_block
