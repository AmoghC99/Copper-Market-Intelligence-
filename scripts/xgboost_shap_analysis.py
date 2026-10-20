from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FEATURES_PATH = Path('data/processed/features_daily.parquet')
OUT_DIR = Path('docs/figures')
SAMPLE_SIZE = 200
RANDOM_STATE = 0


def shap_for_models():
    print('shap: starting')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('shap: reading features parquet', FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    # use same numeric feature selection as model training
    try:
        from copper_market_intelligence.forecasting.models import (
            _numeric_feature_columns,
        )

        feats = _numeric_feature_columns(df)
        X_full = df[feats].copy()
    except Exception:
        X_full = df.select_dtypes(include=[np.number]).copy()

    print('shap: prepared feature frame, nrows=', len(X_full))
    # reproducible sample for SHAP (keep index for local explanations)
    if len(X_full) > SAMPLE_SIZE:
        X = X_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        X = X_full.copy()
    print('shap: sample size used=', len(X))

    import xgboost as xgb

    # keep an untouched sample to reuse per-model
    X_sample = X.copy()

    model_files = sorted(Path('models').glob('xgb_h*.model'))
    if not model_files:
        print('no xgboost model files found in models/; expected xgb_h{h}.model')
        return
    print('shap: found model files:', model_files)

    for mf in model_files:
        name = mf.stem
        horizon = name.replace('xgb_h', '')
        print('shap: loading model', mf)
        # prefer the sklearn wrapper loader (may be safer for Python-side predict)
        bst = None
        reg = None
        try:
            reg = xgb.XGBRegressor()
            reg.load_model(str(mf))
            bst = reg
        except Exception:
            try:
                bst = xgb.Booster()
                bst.load_model(str(mf))
            except Exception as e:
                print('failed loading', mf, e)
                continue

        # feature name verification: try to read stored feature names from model
        model_feat_names = None
        try:
            if reg is not None:
                model_feat_names = getattr(reg, 'feature_names_in_', None)
            if model_feat_names is None:
                # try booster
                model_feat_names = getattr(bst, 'feature_names', None)
        except Exception:
            model_feat_names = None

        # work on a per-model copy so we don't mutate the shared sample
        X_model = X_sample.copy()

        # if model declares feature names, align X_model to that exact set (drop extra cols)
        if model_feat_names is not None:
            model_feat_names = list(model_feat_names)
            # warn and drop any columns not in model features
            missing = [c for c in model_feat_names if c not in X_model.columns]
            extra = [c for c in X_model.columns if c not in model_feat_names]
            if missing:
                print('training data did not have the following fields:', ', '.join(missing))
            if extra:
                print('dropping extra numeric columns not used in model:', extra)
                X_model = X_model[[c for c in X_model.columns if c in model_feat_names]]
            # reorder columns to model order if possible
            X_model = X_model.reindex(columns=[c for c in model_feat_names if c in X_model.columns])

        import traceback

        print('shap: creating explainer (prefer sklearn wrapper, fallback to booster/callable)')
        shap_values = None
        try:
            if reg is not None:
                # use TreeExplainer on sklearn wrapper
                explainer = shap.TreeExplainer(reg)
                shap_values = explainer(X_model)
            else:
                explainer = shap.TreeExplainer(bst)
                shap_values = explainer(X_model)
        except BaseException:
            traceback.print_exc()
            print('TreeExplainer failed; trying callable fallback')
            try:
                if reg is not None:
                    def _predict(arr):
                        return reg.predict(arr)
                else:
                    def _predict(arr):
                        return bst.predict(xgb.DMatrix(arr))
                explainer = shap.Explainer(
                    _predict,
                    X_model,
                    feature_names=X_model.columns.tolist(),
                    max_evals=1000,
                )
                shap_values = explainer(X_model)
            except Exception:
                traceback.print_exc()
                print('callable shap.Explainer fallback also failed for', mf)
                continue

        # compute mean abs shap
        if hasattr(shap_values, 'values'):
            vals = shap_values.values
        else:
            vals = np.array(shap_values)
        mean_abs = np.abs(vals).mean(axis=0)
        feat_names = X_model.columns.tolist()
        # verify exact correspondence between model and training features
        feat_mismatch = None
        if model_feat_names is not None:
            if list(model_feat_names) != feat_names:
                feat_mismatch = {
                    'model_features': list(model_feat_names),
                    'training_features': feat_names,
                }

        meta_out = {
            'model_file': str(mf),
            'horizon': horizon,
            'n_sample': len(X_model),
            'feature_mismatch': feat_mismatch,
        }
        pd.DataFrame([meta_out]).to_csv(OUT_DIR / f'xgb_shap_meta_h{horizon}.csv', index=False)

        imp = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': mean_abs})
        imp = imp.sort_values('mean_abs_shap', ascending=False)
        imp.to_csv(OUT_DIR / f'xgb_shap_feature_importance_h{horizon}.csv', index=False)

        # summary plot
        plt.figure(figsize=(8, 6))
        try:
            shap.summary_plot(shap_values, X_model, show=False)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f'xgb_shap_summary_h{horizon}.png', dpi=150)
            plt.close()
        except Exception as e:
            print('failed to create summary plot for', mf, e)

        # local explanations for first 3 rows
        for i in range(min(3, len(X_model))):
            plt.figure()
            try:
                shap.plots.waterfall(shap_values[i], show=False)
                plt.tight_layout()
                plt.savefig(OUT_DIR / f'xgb_shap_local_h{horizon}_{i}.png', dpi=150)
                plt.close()
            except Exception as e:
                print('failed local shap plot', mf, i, e)

        print('wrote shap outputs for', mf, '->', OUT_DIR)


if __name__ == '__main__':
    import traceback

    try:
        shap_for_models()
    except Exception:
        traceback.print_exc()
        raise
