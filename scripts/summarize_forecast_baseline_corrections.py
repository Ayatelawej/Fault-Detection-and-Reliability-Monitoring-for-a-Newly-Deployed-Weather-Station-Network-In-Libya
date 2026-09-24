"""Verify saved regression predictions and summarise validation-only choices."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.availability.risk_eval import regression_metrics

SOURCES=['forecast_baseline_corrections_20260918',
         'forecast_baseline_corrections_long_20260919',
         'forecast_baseline_corrections_remaining_20260919']
LABELS={'roll_forward':'Roll-forward','persistence':'Persistence','trend':'Trend'}


def run():
    frames=[]
    for source in SOURCES:
        folder=ROOT/'data/eval'/source
        metrics=pd.read_csv(folder/'metrics.csv')
        grid=pd.read_csv(folder/'validation_alpha_grid.csv')
        predictions=pd.read_parquet(folder/'predictions.parquet')
        for i,row in metrics.iterrows():
            choices=grid[(grid.horizon==row.horizon)&(grid.baseline==row.baseline)]
            choice=choices.sort_values(['mae','alpha']).iloc[0]
            assert row.alpha==choice.alpha
            for name in ['mae','rmse','r2']:
                metrics.loc[i,'validation_'+name]=choice[name]
            p=predictions[(predictions.horizon==row.horizon)&(predictions.baseline==row.baseline)]
            assert len(p)==row.n and not p.duplicated(['station_id','hour']).any()
            calculated=regression_metrics(p.actual.to_numpy(),p.predicted.to_numpy())
            np.testing.assert_allclose([row[k] for k in ['mae','rmse','r2']],
                [calculated[k] for k in ['mae','rmse','r2']],rtol=1e-9,atol=1e-9)
        frames.append(metrics)
    result=pd.concat(frames,ignore_index=True).sort_values(['horizon','baseline'])
    assert len(result)==33 and not result.duplicated(['horizon','baseline']).any()
    winners=result.sort_values(['validation_mae','baseline']).groupby('horizon').head(1).sort_values('horizon')
    selected=dict(zip(winners.horizon,winners.baseline))
    result['selected_by_validation']=[selected[h]==b for h,b in zip(result.horizon,result.baseline)]
    out=ROOT/'data/eval/forecast_baseline_corrections_complete_20260919'
    out.mkdir(exist_ok=True)
    result.to_csv(out/'comparison.csv',index=False)
    winners.to_csv(out/'validation_selected.csv',index=False)
    lines=['# Health-score regression: alternative correction baselines','',
        'All three methods predict continuous 0–100 health scores using a baseline plus a learned regression correction: CatBoost except at 144h, where the saved model uses HGB. No classification metric is used here.', '',
        'At each horizon, the feature set, iterations and recency settings match the existing model. Each baseline has a separately trained residual target; validation MAE selects correction strength. Training plus validation is then used for refitting. All original split digests match, and roll-forward control predictions reproduce the saved models. Test metrics were independently recalculated from saved predictions.', '',
        'This is an exploratory extension on previously examined test periods. Settings were originally selected for roll-forward; this is not a separate full hyperparameter search for each baseline. Metrics cover transmitting-origin station-hours. The temporal split is shared by methods at each horizon, but its exact dates and population vary across horizons.', '',
        'Each table cell is **MAE / RMSE / R²**. MAE and RMSE are health-score points (lower is better); R² is dimensionless (higher is better). Bold marks the method selected by validation MAE, including in the test table.', '']
    for title,prefix in [('Validation','validation_'),('Test','')]:
        lines += ['## '+title,'','| Horizon | Roll-forward + model | Persistence + model | Trend + model |','|---|---:|---:|---:|']
        for h,g in result.groupby('horizon'):
            cells=[]
            for baseline in LABELS:
                row=g[g.baseline==baseline].iloc[0]
                text=f"{row[prefix+'mae']:.2f} / {row[prefix+'rmse']:.2f} / {row[prefix+'r2']:.3f}"
                cells.append('**'+text+'**' if baseline==selected[h] else text)
            lines.append('| '+str(h)+'h | '+' | '.join(cells)+' |')
        lines+=['']
    lines+=['## Validation choices','']
    for baseline in LABELS:
        hs=[str(h)+'h' for h,b in selected.items() if b==baseline]
        if hs: lines.append('- '+LABELS[baseline]+': '+', '.join(hs)+'.')
    lines+=['','## Explanation for the report','',
        '> The forecasting model predicts a continuous health score. Earlier summaries emphasised accuracy after converting those predictions into health bands, which did not directly measure numerical forecast error. I have revised the evaluation to report MAE, RMSE and R², and compared three baseline-plus-regression formulations using validation MAE for selection. The underlying task remains health-score regression.', '',
        'No production model or dashboard was changed by this comparison.']
    (ROOT/'docs/health_forecast_baseline_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    run()
