from numba import ir, errors
from . import register_rewrite, ExprRewrite


@register_rewrite('before-inference')
class DetectStaticBinops(ExprRewrite):
    """
    Detect constant arguments to select binops.
    """
    rewrite_expr_of = 'binop'

    # Those operators can benefit from a constant-inferred argument
    rhs_operators = {'**'}

    def match_expr(self, interp, expr, typemap, calltypes):
        # Find binop expressions with a constant rhs
        try:
            if (expr.fn in self.rhs_operators
                and expr.static_rhs is ir.UNDEFINED):
                return interp.infer_constant(expr.rhs)
        except errors.ConstantInferenceError:
            pass

    def rewrite_expr(self, expr, rhs):
        """
        Store constant arguments that were detected in match().
        """
        new_expr = expr.copy()
        new_expr.static_rhs = expr.rhs
        return new_expr
