from numba import ir
from . import register_rewrite, InstRewrite


@register_rewrite('before-inference')
class RewriteConstRaises(InstRewrite):
    """
    Rewrite IR statements of the kind `raise(value)`
    where `value` is the result of instantiating an exception with
    constant arguments
    into `static_raise(exception_type, constant args)`.

    This allows lowering in nopython mode, where one can't instantiate
    exception instances from runtime data.
    """

    rewrite_inst_of = ir.Raise

    def _is_exception_type(self, const):
        return isinstance(const, type) and issubclass(const, Exception)

    def _break_constant(self, interp, const):
        """
        Break down constant exception.
        """
        if isinstance(const, BaseException):
            return const.__class__, const.args
        elif self._is_exception_type(const):
            return const, None
        else:
            raise NotImplementedError("unsupported exception constant %r"
                                      % (const,))

    def match_inst(self, interp, inst, typemap, calltypes):
        if inst.exception is None:
            # re-reraise
            exc_type, exc_args = None, None
        else:
            # raise <something> => find the definition site for <something>
            const = interp.infer_constant(inst.exception)
            exc_type, exc_args = self._break_constant(interp, const)
        return exc_type, exc_args

    def rewrite_inst(self, inst, data):
        """
        Rewrite all matching raise as static_raise.
        """
        exc_type, exc_args = data
        return ir.StaticRaise(exc_type, exc_args, inst.loc)


