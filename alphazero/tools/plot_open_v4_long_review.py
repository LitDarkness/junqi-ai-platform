"""Render frozen recorded development results; does not touch training."""
from pathlib import Path
import json


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    root=Path(__file__).resolve().parents[1]
    out=root/'reports/current_training_audit_20261004'
    arena=json.loads((root/'runs/open_junqi_v4_tactical_training_20261004_065605/arena/block_000006_screen.json').read_text())
    available={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family']=next((name for name in ('Microsoft YaHei','SimHei','Noto Sans CJK SC') if name in available),'DejaVu Sans')
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10.8,5.7),dpi=160)
    names=['champion','parent','control','candidate'];labels=['正式26','来源32','对照38','根过滤候选38']
    intervals=[next(r['family_score'] for r in arena['summaries'] if r['candidate']==n and r['opponent']=='teacher0') for n in names]
    colors=['#64748b','#475569','#2563eb','#0d9488']
    for i,r in enumerate(intervals):
        y=r['mean']*100;lo,hi=np.array(r['ci95'])*100
        ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',markersize=10,capsize=7,color=colors[i],linewidth=2)
        ax.annotate(f'{y:.2f}%\n[{lo:.2f}, {hi:.2f}]',(i,hi),xytext=(0,13),textcoords='offset points',ha='center',fontsize=10)
    ax.axhline(50,color='#9ca3af',linestyle='--',linewidth=1.2)
    ax.text(3.43,51,'50%得分',ha='right',color='#6b7280',fontsize=9)
    ax.set_xticks(range(4),labels);ax.set_ylim(0,68);ax.set_xlim(-.45,3.5)
    ax.set_ylabel('对冻结强启发式082的家族平均得分（%）')
    ax.set_title('当前已完成开发筛查：点估计上升，根过滤增益尚未确认',loc='left',fontsize=14,pad=24)
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.text(.11,.035,'固定16探索家族 / 95%家族bootstrap区间；旧动态批量存在重复基线漂移。\n082/084完整棋路等价；该图不证明泛化、晋级或单调长期提升。',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(out/'long_term_scores.png');fig.savefig(out/'long_term_scores.svg');plt.close(fig)
    print(out/'long_term_scores.png')


if __name__=='__main__':main()
