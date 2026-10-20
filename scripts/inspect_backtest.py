import importlib
import inspect
import textwrap

mod = importlib.import_module('copper_market_intelligence.forecasting.backtest')
fn = getattr(mod, 'expanding_walk_forward_with_meta')
print('signature:', inspect.signature(fn))
print('\nsource:\n')
src = inspect.getsource(fn)
print(textwrap.indent(src, '    '))
