from numba import ir, errors
from . import register_rewrite, Rewrite


@register_rewrite('before-inference')
class RewriteConstGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$constXX)`
    where `$constXX` is a known constant as
    `static_getitem(value=arr, index=<constant value>)`.
    """

    def match(self, interp, block, typemap, calltypes):
        self.getitems = getitems = {}
        self.block = block
        # Detect all getitem expressions and find which ones can be
        # rewritten
        for expr in block.find_exprs(op='getitem'):
            try:
                const = interp.infer_constant(expr.index)
            except errors.ConstantInferenceError:
                continue
            getitems[expr] = const

        return len(getitems) > 0

    def apply(self):
        """
        Rewrite all matching getitems as static_getitems.
        """
        new_block = self.block.copy()
        new_block.clear()

        for m, inst, expr in self.block.match_exprs(op='getitem'):
            new_inst = inst.copy()
            if m and expr in self.getitems:
                const = self.getitems[expr]
                new_expr = ir.Expr.static_getitem(value=expr.value,
                                                  index=const,
                                                  index_var=expr.index,
                                                  loc=expr.loc)
                new_inst.value = new_expr
            new_block.append(new_inst)
        return new_block


@register_rewrite('before-inference')
class RewriteConstSetitems(Rewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """

    def match(self, interp, block, typemap, calltypes):
        self.setitems = setitems = {}
        self.block = block
        # Detect all setitem statements and find which ones can be
        # rewritten
        for inst in block.find_insts(ir.SetItem):
            try:
                const = interp.infer_constant(inst.index)
            except errors.ConstantInferenceError:
                continue
            setitems[inst] = const

        return len(setitems) > 0

    def apply(self):
        """
        Rewrite all matching setitems as static_setitems.
        """
        new_block = self.block.copy()
        new_block.clear()
        for m, inst in self.block.match_insts(ir.SetItem):
            new_inst = inst.copy()
            if m and inst in self.setitems:
                const = self.setitems[inst]
                new_inst = ir.StaticSetItem(inst.target, const,
                                            inst.index, inst.value, inst.loc)
            new_block.append(new_inst)
        return new_block
