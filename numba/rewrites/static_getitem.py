from numba import ir, errors
from . import register_rewrite, InstRewrite, ExprRewrite


@register_rewrite('before-inference')
class RewriteConstGetitems(ExprRewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$constXX)`
    where `$constXX` is a known constant as
    `static_getitem(value=arr, index=<constant value>)`.
    """
    rewrite_expr_of = 'getitem'

    def match_expr(self, interp, expr, typemap, calltypes):
        try:
            return interp.infer_constant(expr.index)
        except errors.ConstantInferenceError:
            pass

    def rewrite_expr(self, expr, const):
        """
        Rewrite all matching getitems as static_getitems.
        """
        return ir.Expr.static_getitem(value=expr.value, index=const,
                                      index_var=expr.index, loc=expr.loc)


@register_rewrite('before-inference')
class RewriteConstSetitems(InstRewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """

    rewrite_inst_of = ir.SetItem

    def match_inst(self, interp, inst, typemap, calltypes):
        # Detect all setitem statements and find which ones can be
        # rewritten
        try:
            return interp.infer_constant(inst.index)
        except errors.ConstantInferenceError:
            pass

    def rewrite_inst(self, inst, const):
        """
        Rewrite all matching setitems as static_setitems.
        """
        return ir.StaticSetItem(inst.target, const, inst.index, inst.value,
                                inst.loc)
