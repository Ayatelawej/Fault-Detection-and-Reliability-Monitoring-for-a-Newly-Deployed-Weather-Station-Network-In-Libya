"""Historical HGB window-length comparison; missing temporal runs remain missing."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
    root=Path(__file__).resolve().parents[1]
    data=pd.read_csv(root/'data/report/hgb_history_window_comparison.csv')
    temporal=pd.read_csv(root/'data/eval/blocked_model_comparison_20260916_v2/comparison.csv').set_index('model').loc['HGB']
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),sharey=True)
    colors=['#295A7B','#32715C']
    for ax,split,title in zip(axes,['random','spaced','blocked'],['Random partition','Spaced partition','Temporal holdout']):
        if split!='blocked':
            part=data[data.split_scheme.eq(split)].sort_values('window_hours')
            for metric,label,color in zip(['validation_f1','test_f1'],['Validation','Development test'],colors):
                ax.plot(part.window_hours,100*part[metric],'-o',color=color,label=label,linewidth=1.7,markersize=5)
        else:
            for metric,label,color in zip(['validation_f1','test_f1'],['Validation','Development test'],colors):
                value=100*temporal[metric]
                ax.plot([1],[value],'o',color=color,label=label,markersize=6)
                ax.annotate(f'{value:.2f}%',(1,value),xytext=(9,0),textcoords='offset points',va='center',fontsize=10,color=color)
            ax.text(14,91.4,'3–24-hour windows\nnot evaluated',ha='center',va='center',fontsize=11,color='#53616C')
        ax.set_title(title,fontsize=13)
        ax.set_xlim(-.2,25.3); ax.set_ylim(84,95)
        ax.set_xticks([1,3,5,7,12,24]); ax.set_yticks([84,86,88,90,92,94])
        ax.set_xlabel('Input window (hours)',fontsize=11)
        ax.grid(color='#E4E7EA',linewidth=.7)
        ax.set_axisbelow(True)
        for side in ('right','top'): ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Fault-class F1 (%)',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle('HGB performance by input-window length',fontsize=16,y=.99)
    fig.text(.5,.025,'Historical results using the original feature pipeline, including retrospective stuck indicators.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.065,right=.99,top=.73,bottom=.19,wspace=.12)
    out=root/'docs/figures/hgb_window_comparison_with_temporal.png'
    fig.savefig(out,dpi=300,facecolor='white'); plt.close(fig)
    print(out)

if __name__=='__main__': main()
