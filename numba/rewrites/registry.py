from __future__ import print_function, division, absolute_import

from collections import defaultdict
import sys

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

    def apply(self):
        '''Overload this method to return a rewritten IR basic block when a
        match has been found.
        '''
        raise NotImplementedError("Abstract Rewrite.apply() called!")


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

    def apply(self):
        block = self.block
        new_block = block.copy()
        new_block.clear()
        for inst in block.body:
            data = self._matched.get(inst)
            new_inst = (inst if data is None else self.rewrite_inst(inst, data))
            new_block.append(new_inst)
        return new_block

    def match_inst(self, inst):
        raise NotImplementedError("Abstract InstRewrite.match_inst() called!")

    def rewrite_inst(self, inst, data):
        raise NotImplementedError("Abstract InstRewrite.rewrite_inst() called!")


class ExprRewrite(Rewrite):
    '''
    Simple expression rewriter
    '''
    rewrite_expr_of = None

    def match(self, interp, block, typmap, calltypes):
        self._matched = matched = {}
        self.block = block
        for m, inst, expr in block.match_exprs(self.rewrite_expr_of):
            if m:
                result = self.match_expr(interp, expr, typmap, calltypes)
                if result is not None:
                    matched[inst] = result

        return matched

    def apply(self):
        block = self.block
        new_block = block.copy()
        new_block.clear()
        for m, inst, expr in block.match_exprs(self.rewrite_expr_of):
            if m and inst in self._matched:
                data = self._matched.get(inst)
                new_expr = (expr if data is None
                            else self.rewrite_expr(expr, data))
                new_inst = inst.copy()
                new_inst.value = new_expr
                new_block.append(new_inst)
            else:
                new_block.append(inst)

        return new_block

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
                    new_block = rewrite.apply()
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
