"""Generate report-ready validation, test and July tables with explicit scopes."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/eval/final_system_release_20260924'


def table(frame,columns,labels=None,percent=(),digits=2):
    labels=labels or columns
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        values=[]
        for col in columns:
            value=row[col]
            if pd.isna(value):value='—'
            elif col in percent:value=f'{100*float(value):.{digits}f}%'
            elif isinstance(value,float):value=f'{value:.{digits}f}'
            else:value=str(value)
            values.append(value)
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def run():
    random=pd.read_csv(ROOT/'data/report/one_hour_detection_model_comparison.csv');random=random[random.split.eq('random')].copy()
    temporal=pd.read_csv(ROOT/'data/eval/blocked_model_comparison_20260916_v2/comparison.csv');temporal['split']='temporal'
    binary=pd.concat([random,temporal],ignore_index=True)
    reason=pd.read_csv(ROOT/'data/eval/final_reason_comparison_20260924/summary.csv')
    forecast=pd.read_csv(ROOT/'data/eval/final_forecast_comparison_20260924/metrics.csv')
    binary.to_csv(OUT/'binary_development_tables.csv',index=False)
    reason.to_csv(OUT/'reason_development_tables.csv',index=False)
    forecast.to_csv(OUT/'forecast_development_tables.csv',index=False)
    chosen=pd.read_csv(ROOT/'data/eval/final_forecast_comparison_20260924/chosen.csv')
    pieces=['# Final model comparison and selected release',
        'Completed comparisons for the main development dataset. Validation tables come first; test tables are reporting only. Logistic experiments remain archived and are not part of the requested three-model classification comparison.',
        '## Selection and interpretation',
        '- Binary: maximise the minimum of temporal validation precision, recall and F1.\n- Mechanisms and components: maximise temporal validation micro-F1 separately; minimum-one output on both axes, original episode reasons, all reference fault hours.\n- Forecasts: minimise temporal validation MAE per horizon. Band accuracy is a separate secondary metric, never the selection criterion.\n- Stratified random holdout preserves binary class proportions. Blocked temporal classification uses February validation and March–April testing, with training on both sides. Forecasting uses its original horizon-purged chronological partitions, not the classification months.\n- Normalised regression score (%) = 100 − MAE on the 0–100 health scale. This is not the percentage of correct forecasts. Band accuracy is the percentage in the correct band (<40, 40–<60, 60–<80, >=80).',
        '## Validation results']
    for part in ('validation','test'):
        if part=='test':pieces.append('## Test results — not used for selection')
        pieces.append('### Binary fault detection')
        for split,title in [('random','Stratified random holdout'),('temporal','Blocked temporal holdout')]:
            f=binary[binary.split.eq(split)].set_index('model').loc[['HGB','EF-HGB','RGFN']].reset_index()
            cols=[f'{part}_{k}' for k in ('accuracy','precision','recall','f1')]
            pieces.extend([f'#### {title}',table(f,['model']+cols,['Model','Accuracy','Precision','Recall','F1'],percent=cols)])
        pieces.append('### Mechanism and component classification')
        for split,title in [('random','Stratified random holdout'),('temporal','Blocked temporal holdout')]:
            for axis in ('mechanism','component'):
                f=reason[reason.split.eq(split)&reason.partition.eq(part)&reason.axis.eq(axis)].set_index('model').loc[['HGB','EF-HGB','RGFN']].reset_index()
                cols=['micro_precision','micro_recall','micro_f1','macro_f1']
                pieces.extend([f'#### {title}: {axis}',table(f,['model','rows']+cols,['Model','Fault hours','Micro precision','Micro recall','Micro-F1','Macro-F1'],percent=cols)])
        pieces.append('### Health forecasting — transmitting-origin hours')
        f=forecast[forecast.partition.eq(part)].copy();f['R²']=f.r2.map(lambda v:f'{v:.3f}')
        pieces.append(table(f,['horizon_h','model','mae','rmse','R²','normalized_score_pct','band_accuracy_pct'],
            ['Horizon (h)','Model','MAE','RMSE','R²','Normalised score (%)','Band accuracy (%)']))
    pieces.extend(['## Validation-selected system',
        'Binary: **EF-HGB**, threshold **0.50**. Mechanism: **EF-HGB**. Component: **EF-HGB**. Both reason outputs require at least one code.',
        table(chosen,['horizon_h','model','mae'],['Forecast horizon (h)','Selected model','Validation MAE']),
        'The release refits selected configurations using eligible development data through June. The tables above belong to the held-out experiment models, not the final refits.'])
    pieces.append('## July evaluation of the selected refits')
    b=json.loads((OUT/'july_binary_metrics.json').read_text());pieces.append('### Binary detector')
    pieces.append(table(pd.DataFrame([b]),['accuracy','precision','recall','f1'],['Accuracy','Precision','Recall','F1'],percent=['accuracy','precision','recall','f1']))
    pieces.append('### Reasons: distinguish the evaluation populations')
    j=pd.read_csv(OUT/'july_reason_metrics.csv')
    pieces.append(table(j,['axis','scope','rows','micro_precision','micro_recall','micro_f1','macro_f1'],
        ['Output','Population','Hours','Micro precision','Micro recall','Micro-F1','Macro-F1'],percent=['micro_precision','micro_recall','micro_f1','macro_f1']))
    pieces.append('All-reference-fault evaluation bypasses the binary gate; correctly-detected-fault evaluation excludes missed faults and false alarms. All-alert evaluation includes false alarms but not missed faults. End-to-end includes both. Original episode reasons cover all 1,701 July reference fault hours: no current-hour-reason exclusions are applied.')
    pieces.append('### Health forecasts')
    f=pd.read_csv(OUT/'july_forecast_metrics.csv');f['R²']=f.r2.map(lambda v:f'{v:.3f}')
    pieces.append(table(f,['horizon_h','model','rows','mae','rmse','R²','normalized_score_pct','band_accuracy_pct'],
        ['Horizon (h)','Model','Hours','MAE','RMSE','R²','Normalised score (%)','Band accuracy (%)']))
    pieces.append('Only July origins with an observed future health target are scored. Longer horizons have fewer eligible July origins; unknown August targets are not invented. The dashboard shows 1/3/6/12/24h; selected models and July evaluation extend to 168h.')
    pieces.extend(['## Protocol and limitations',
        'Binary comparisons reuse the historical 67-feature pipeline, including the previously identified retrospective stuck indicators. These results do not repair or establish fully causal binary detection. Binary RGFN results are the saved multi-seed summaries.',
        'Reason RGFN is a new single-seed adaptation of the existing one-hour MLP gated architecture to the same head-specific feature views as HGB/EF-HGB (322-feature library). It is not the retired current-hour-target result. All heads use original episode targets; per-head fusion/thresholds and neural checkpoints use validation, never test. Macro-F1 averages only positive-support classes, whose membership differs between periods.',
        'Forecasts share feature scope, recency settings and baseline at each horizon. HGB/CatBoost search 100/200/300 iterations and direct versus residual prediction; linear/Ridge reproduce the earlier validation-selected grid. This is not an exhaustive feature search. At 1h the linear candidates select zero correction, so their output is the standalone baseline.',
        'These development splits and July have been explored previously. July is a retrospective extension, not a newly untouched external validation. Archived weather-reference arrival-time availability remains unverified. Grouped/spaced results and logistic tests remain archived as supplementary experiments; they were not deleted.',
        'Release paths: `data/model/final_system_20260924/` and `data/eval/final_system_release_20260924/`. Previous releases remain available for rollback. This document supplies replacement report tables; the user’s Word report has not been rewritten.'])
    path=ROOT/'docs/final_model_selection_results.md';path.write_text('\n\n'.join(pieces)+'\n',encoding='utf-8');print(path)

if __name__=='__main__':run()
