from __future__ import print_function, division, absolute_import

from collections import defaultdict
import sys
import types

from numba import config, ir


class Rewrite(object):
    '''Defines the abstract base class for Numba rewrites.
    '''

    def __init__(self, pipeline):
        '''Constructor for the Rewrite class.
        '''
        self.pipeline = pipeline

    def match(self, interp, block, typemap, calltypes):
        '''Overload this method to check an IR block for matching terms in the
        rewrite.
        '''
        return False

    def apply(self, new_block):
        '''Overload this method to return a rewritten IR basic block when a
        match has been found.
        '''
        raise NotImplementedError("Abstract Rewrite.apply() called!")


def _as_generator(val):
    if not isinstance(val, types.GeneratorType):
        return (val,)
    else:
        return val


class InstRewrite(Rewrite):
    '''
    Simple instruction rewriter
    '''
    rewrite_inst_of = None

    def match(self, interp, block, typmap, calltypes):
        assert self.rewrite_inst_of is not None
        self._matched = matched = {}
        self.block = block
        for inst in block.find_insts(self.rewrite_inst_of):
            result = self.match_inst(interp, inst, typmap, calltypes)
            if result is not None:
                matched[inst] = result

        return matched

    def apply(self, new_block):
        block = self.block
        for inst in block.body:
            data = self._matched.get(inst)
            if data is None:
                new_block.append(inst)
            else:
                for new_inst in _as_generator(self.rewrite_inst(inst, data)):
                    new_block.append(new_inst)

    def match_inst(self, inst):
        raise NotImplementedError("Abstract InstRewrite.match_inst() called!")

    def rewrite_inst(self, inst, data):
        raise NotImplementedError("Abstract InstRewrite.rewrite_inst() called!")


class ExprRewrite(InstRewrite):
    '''
    Simple expression rewriter
    '''
    rewrite_expr_of = None

    @property
    def rewrite_inst_of(self):
        return ir.Assign

    def match_inst(self, interp, inst, typmap, calltypes):
        expr = inst.value
        if isinstance(expr, ir.Expr):
            pat = self.rewrite_expr_of
            if pat is not None and pat == expr.op:
                return self.match_expr(interp, expr, typmap, calltypes)

    def rewrite_inst(self, inst, data):
        new_inst = inst.copy()
        expr = inst.value
        new_inst.value = self.rewrite_expr(expr, data)
        return new_inst

    def match_expr(self, expr):
        raise NotImplementedError("Abstract ExprRewrite.match_expr() called!")

    def rewrite_expr(self, expr, data):
        raise NotImplementedError("Abstract ExprRewrite.rewrite_expr() called!")


class RewriteRegistry(object):
    '''Defines a registry for Numba rewrites.
    '''
    _kinds = frozenset(['before-inference', 'after-inference'])

    def __init__(self):
        '''Constructor for the rewrite registry.  Initializes the rewrites
        member to an empty list.
        '''
        self.rewrites = defaultdict(list)

    def register(self, kind):
        """
        Decorator adding a subclass of Rewrite to the registry for
        the given *kind*.
        """
        if not kind in self._kinds:
            raise KeyError("invalid kind %r" % (kind,))
        def do_register(rewrite_cls):
            if not issubclass(rewrite_cls, Rewrite):
                raise TypeError('{0} is not a subclass of Rewrite'.format(
                    rewrite_cls))
            self.rewrites[kind].append(rewrite_cls)
            return rewrite_cls
        return do_register

    def apply(self, kind, pipeline, interp):
        '''Given a pipeline and a dictionary of basic blocks, exhaustively
        attempt to apply all registered rewrites to all basic blocks.
        '''
        assert kind in self._kinds
        blocks = interp.blocks
        old_blocks = blocks.copy()
        for rewrite_cls in self.rewrites[kind]:
            # Exhaustively apply a rewrite until it stops matching.
            rewrite = rewrite_cls(pipeline)
            work_list = list(blocks.items())
            while work_list:
                key, block = work_list.pop()
                matches = rewrite.match(interp, block, pipeline.typemap,
                                        pipeline.calltypes)
                if matches:
                    if config.DEBUG or config.DUMP_IR:
                        print("_" * 70)
                        print("REWRITING (%s):" % rewrite_cls.__name__)
                        block.dump()
                        print("_" * 60)
                    new_block = block.copy()
                    new_block.clear()
                    rewrite.apply(new_block)
                    blocks[key] = new_block
                    work_list.append((key, new_block))
                    if config.DEBUG or config.DUMP_IR:
                        new_block.dump()
                        print("_" * 70)
        # If any blocks were changed, perform a sanity check.
        for key, block in blocks.items():
            if block != old_blocks[key]:
                block.verify()


rewrite_registry = RewriteRegistry()
register_rewrite = rewrite_registry.register
