"""Prepare a fresh, bounded one-learner Policy-step-rate experiment."""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import unittest


def main():
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now
    from open_v4_stable_policy_contract import verify_single_prerequisite
    root=Path(__file__).resolve().parents[1]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def entry(path):
        path=Path(path);return dict(path=str(path),sha256=sha(root/path))
    parser=argparse.ArgumentParser();parser.add_argument('--register',metavar='PLAN');args=parser.parse_args()
    receipt_path=root/'reports/stable_policy_20261005/PREPARATION.json'
    if args.register:
        plan_path=root/args.register;plan=read_json(plan_path)
        verify_single_prerequisite(root,plan)
        check=read_json(root/'reports/stable_policy_20261005/PRELAUNCH_VERIFICATION.json')
        if not check['valid'] or check['plan_sha256']!=sha(plan_path):raise ValueError('successful exact-plan prelaunch needed')
        masters=[root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json',root/'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json']
        master=read_json(masters[0])
        if master['optimization']['type']!='complete_policy_value_output_diagnostic' or master.get('paused_by_user'):raise ValueError('fresh explicit authorization expected')
        master['completed_output_diagnostic']=dict(master['optimization'],state='complete',result=plan['diagnostic'])
        master['optimization']=dict(type=plan['route'],state='registered',config=str(plan_path.relative_to(root)).replace('\\','/'),config_sha256=sha(plan_path),
            run_directory=plan['directory'],status_tool='tools/run_open_v4_stable_policy.py',launcher='start_open_v4_stable_policy.ps1',
            training_variants=['learner'],training_active=False,source_iteration=158,source_model_hash=plan['source_checkpoint']['model_hash'],
            factor='Policy phase rate .25; Value nominal rate unchanged',maximum_stages=24,first_horizon_stages=12,
            new_training_games=0,effective_updates=0,registered_at=utc_now(),screen_games=344,full_gate_games=5184)
        master['automatic_continuation_enabled']=True
        master['next_action']='Launch the registered fresh single Policy-step-rate run now after exact full-state preflight. Sole owner; screen6/12/24, stop on prospective external/direct negative signal. At12 only original full64 confirmed gain permits same-process next12. Unconfirmed12 or24 review_due; no blind continuation. Historical STOPs preserved; new branch STOP and later user pause override.'
        master['reporting_contract']['latest_scheme']='Single Policy-phase LR .25 relative to failed256 update route. Source158 had64updates, so source-to-new changes both exposure and Policy rate. No concurrent causal control. Online fresh games and accepted updates separate from frozen diagnostic/evaluation.'
        for p in masters:atomic_json(p,master)
        docs=root/'docs/LONG_RUNNING_TRAINING_zh-CN.md'
        text=docs.read_text(encoding='utf-8');heading,rest=text.split('\n',1)
        addition='''

## 2026年10月5日：诊断后登记新单支Policy步幅实验

最新用户“找到可行的方案就继续训练”授权立即启动登记的新独立分支，覆盖下方旧暂停期限制。旧182失败分支、全部旧STOP、CURRENT角色、原始分片和阴性结果保持。16个保存训练留出根的完整Policy×Value诊断共4096模拟，零新局/零更新；9个根推荐改变，其中保留旧158 Policy恢复7个、保留旧Value恢复2个。只支持检查Policy先验漂移，不证明旧动作更优或已定位唯一根因。

新配置`CONFIG`，目录`DIRECTORY`，入口`tools/run_open_v4_stable_policy.py`。从受保护完整158模型、Adam矩、步数、调度、计数、全部RNG和Value尺度派生一个Learner。保持失败256更新路线的新数据覆盖机会，每段自身前段完整Learner产生256真实新终局，再尝试256 PPV更新；仅Policy阶段学习率乘0.25，Policy共享编码器3.125e-6、Policy头1.25e-5；Value编码器5e-6、WDL头5e-5保持，回退规则不变。相对来源158还改变64→256更新数，不能称严格单因素因果或4×0.25等价。保存原25/25/50混合监督、真实自博弈访问、教师硬目标、真实终局WDL、三段回放、64/128搜索、全程噪声、24ply后greedy、k2强制根探索；修剪/guard/dedup关闭。

初定12段，3072新局/3072计划更新；第6/12固定16研究筛查344局，只有第12正信号才原完整64门禁5184局。负面早停使用最多6/12/24三次固定观察、两个指标的99.1667%家族bootstrap区间，8000重采样：外部相对来源点差≤−5pp且区间上界<0，或直接对来源点值≤45%且区间上界<50%，任一停止复盘。bootstrap不是有限样本保证。正式采用仍要求四组保护参照的两个相关教师配对95%下界>0、四项网络直接区间下界>50%、来源复测差≤2pp和原Champion完整门禁；固定16不晋级。第12完整门禁确认才同一独占进程即时接下12段，最多24；未确认不盲延长。开发反复使用、082/084相关，不能当独立最终证据，最终布局不使用。较小筛查减少无关保护比赛，完整正式门禁不缩减；真实吞吐和墙钟另记，不宣称倍数加速。

只有一个GPU训练器，没有等待控制器。持锁推进只读监督，不修改冻结源码、配置或二进制、不启动第二任务。STOP优先，新分支两处STOP及PLAN.paused_by_user生效，历史STOP保持归档保护。定时ai待首段真实数据/完整状态核验后恢复，每30分钟静默跟随最新PLAN，仅实质收益、晋级、异常或需要处理时通知。在线训练=自身前段Learner新局+学习；旧重放=离线；本次诊断和冻结比赛=评测，不混记轮数。新的159/170/可能182都属于新分支，不覆盖失败旧170/182。
'''.replace('CONFIG',str(plan_path.relative_to(root)).replace('\\','/')).replace('DIRECTORY',plan['directory'])
        docs.write_text(heading+addition+'\n'+rest,encoding='utf-8')
        atomic_json(root/'reports/stable_policy_20261005/REGISTRATION.json',dict(at=utc_now(),plan=entry(plan_path.relative_to(root)),master_digest=sha(masters[0]),old_STOPs_preserved=True))
        print(json.dumps(dict(registered=True,config=str(plan_path.relative_to(root)),directory=plan['directory']),ensure_ascii=False));return
    if receipt_path.exists():raise ValueError('already prepared; do not silently overwrite registered evidence')
    torch.set_num_threads(2)
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_open_v4_stable_policy.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    validation=dict(at=utc_now(),all_passed=result.wasSuccessful() and result.testsRun==4 and not result.skipped,
        tests_run=result.testsRun,skipped=result.skipped,output=stream.getvalue(),cpu_no_gpu_initialized=not torch.cuda.is_initialized(),
        test_source=entry('tests/test_open_v4_stable_policy.py'))
    cpu_stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    cpu_path=Path('reports/stable_policy_20261005/CPU_VALIDATION_'+cpu_stamp+'.json');atomic_json(root/cpu_path,validation)
    if not validation['all_passed'] or not validation['cpu_no_gpu_initialized']:raise ValueError('CPU validation failed')
    launcher=root/'start_open_v4_stable_policy.ps1'
    if launcher.exists():raise ValueError('fresh launcher expected')
    original=(root/'start_open_v4_single_ratio.ps1').read_text(encoding='utf-8')
    launcher.write_text(original.replace('single_ratio','stable_policy'),encoding='utf-8')
    old_path=Path('config/experiments/open_v4_single_ratio_20261005_070831.json');old=read_json(root/old_path)
    source=torch.load(root/old['source_checkpoint']['path'],map_location='cpu',weights_only=False)
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    directory='runs/open_junqi_v4_stable_policy_'+stamp
    plan_path=Path('config/experiments/open_v4_stable_policy_'+stamp+'.json')
    base_path=Path('config/experiments/open_v4_stable_policy_base_'+stamp+'.json')
    cfg=copy.deepcopy(source['config']);cfg['policy_step_lr_multiplier']=.25;atomic_json(root/base_path,cfg)
    plan=copy.deepcopy(old)
    plan.update(schema='junqi-stable-policy-plan-1',route='single_policy_step_rate',registered_at=utc_now(),directory=directory,
        parent_config=str(base_path),parent_config_sha256=sha(root/base_path),policy_step_lr_multiplier=.25,
        maximum_stages=24,segment_stages=12,stages_per_block=6,screen_boundaries=[6,12,24],
        small_screen_matchups=[['candidate','parent'],['parent','teacher0'],['candidate','teacher0'],['repeat','teacher0']],
        screen_games=344,full_games=5184,evaluation_matchups=16,small_evaluation_matchups=4,
        registered_first_segment_new_games=3072,registered_first_segment_updates=3072,registered_maximum_new_games=6144,registered_maximum_updates=6144)
    plan['early_stop']=dict(confidence=1-.05/6,paired_delta_limit=-.05,direct_mean_limit=.45,paired_upper_limit=0.,direct_upper_limit=.5,
        maximum_observations=3,endpoints=2,bootstrap_resamples=8000,source_drift_tolerance=.02,
        explanation='Prospective6/12/24 only; Bonferroni allocation for two endpoints times three looks; family bootstrap is not a finite-sample guaranteed sequential test.')
    plan['stop_paths']=[directory+'/STOP',directory+'/supervision/STOP']
    pause=read_json(root/'reports/training_pause_182_20261005/PAUSE_VERIFICATION.json')
    paths={e['path'].replace('\\','/') for e in plan['historical_stops']+pause['stop_markers']}
    paths.update(p.as_posix() for p in Path('runs/open_junqi_v4_single_20261005_044709').glob('**/STOP'))
    plan['historical_stops']=[entry(p) for p in sorted(paths)]
    plan['historical_stop_markers']=sorted(paths)
    failed_dir=old['directory'];plan['frozen_runs'].append(dict(directory=failed_dir,current_sha256=sha(root/failed_dir/'CURRENT.json')))
    plan['diagnostic']=entry('runs/open_junqi_v4_output_diagnostic_20261005_103918/RESULTS.json')
    plan['prior_negative_review']=entry('runs/open_junqi_v4_single_ratio_20261005_070831/supervision/REVIEWS/FINAL_PAUSED_REVIEW_20261005.json')
    plan['cpu_verification']=entry(cpu_path)
    plan['previous_experiment']=dict(plan=entry(old_path),review=plan['prior_negative_review'],interpretation='Stopped negative256-update route, never resumed.')
    plan['authorization']='Latest user: 找到可行的方案就继续训练. New registered one-learner experiment authorized; old runs/STOPs never resumed or cleared.'
    plan['source_role']='Protected158 complete research state; failed182 archived. Source158 has no formal external qualification.'
    plan['interpretation']='Relative to failed256-update route, only Policy-phase LR is quartered. Relative to source158 both updates64to256 and Policy LR differ. Adam moments, scheduler and RNG inherited; phase current LR intentionally migrated. One evolving learner, no simultaneous causal control, no guaranteed strength gain.'
    plan['hypothesis']='Complete-output16-root diagnosis:9 recommendations change; old Policy recovers7, old Value2. Investigate Policy-prior drift by slowing Policy steps while preserving exposure and nominal Value LR. Recovery is not proof that old actions are better or unique root cause.'
    plan['horizon']='First12 stages158to170,3072fresh/3072planned PPV updates.6negative may stop;12unconfirmed review_due.Only original full64 qualified gain permits same-process next12, maximum24 new branch182. No18screen, no blind extension.'
    plan['efficiency']='Four-matchup344 research screens instead of16-matchup1376; original full64 gate5184 unchanged. This reduces screen work, not demonstrated total wall-clock speedup; collection/learning/search costs measured.'
    plan['promotion_rule']='Original full64 Champion gate unchanged PLUS four groups Champion/segment source/frozen candidate134/control86: each of two correlated teacher paired95lower>0, four network direct95lower>.5, repeat-source point drift<=.02. Fixed16 never promotes, final untouched.'
    additions=['python/junqi_open_v4/stable_policy_config.py','python/junqi_open_v4/stable_policy_learner.py','tools/open_v4_stable_policy_contract.py',
        'tools/run_open_v4_stable_policy.py','tools/prepare_open_v4_stable_policy.py','tools/audit_open_v4_stable_startup.py',
        'tools/diagnose_open_v4_policy_value.py','tests/test_open_v4_stable_policy.py','start_open_v4_stable_policy.ps1']
    for p in additions:plan['source'][p]=sha(root/p)
    # Validate every existing frozen dependency now, never refresh changed old hashes.
    for p,h in old['source'].items():
        if sha(root/p)!=h:raise ValueError('historical executed source changed: '+p)
    atomic_json(root/plan_path,plan)
    verify_single_prerequisite(root,plan)
    atomic_json(receipt_path,dict(at=utc_now(),plan=entry(plan_path),directory=directory,cpu_validation=entry(cpu_path),diagnostic=plan['diagnostic'],
        historical_stops=len(plan['historical_stops']),source_model_hash=source['model_hash'],old_failed182_preserved=True))
    print(json.dumps(dict(prepared=True,config=str(plan_path),directory=directory,tests_passed=result.testsRun),ensure_ascii=False))


if __name__=='__main__':main()
