import runpy
import traceback

try:
    runpy.run_path('scripts/xgboost_shap_analysis.py', run_name='__main__')
except Exception as e:
    traceback.print_exc()
    print('ERROR:', e)
