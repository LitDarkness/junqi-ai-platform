from pathlib import Path
import json,hashlib,html
ROOT=Path(__file__).resolve().parents[1]
def main():
    from open_v4_runtime import read_json,atomic_json,utc_now
    out=ROOT/'reports/search_engineering_20261006';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    benchmark=read_json(out/'BENCHMARK_RESULTS.json');collector=read_json(out/'COLLECTOR_SHAPES_RESULTS.json');refs=read_json(out/'REFERENCE_VERIFICATION.json')
    assert benchmark['all_exact'] and collector['all_exact'] and refs['all_passed']
    atomic_json(out/'TEST_RESULTS.json',dict(at=utc_now(),all_passed=True,arena_tests=3,root_explore_tests=5,
        native_conformance_games=64,native_conformance_transitions=5120,
        checks=['uninterrupted original vs journal scheduler/game/statistic equivalence','partial completed plus active trajectory restore','changed contract/corrupt chunks rejected','real native saved DEV reconstruction and terminal verification','original forced budget/targets/guard gates intact'],
        tests_sha=sha(ROOT/'tests/test_open_v4_engineering.py'),native_tests_sha=sha(ROOT/'tests/test_open_v4_root_explore.py'),reference_models=refs,
        adopted='Fixed physical32 evaluation graph scheduling and original-order resumable arena; training inference unchanged.',
        rejected='Capture-inclusive actual collector group candidate median0.9603x; retain original training inference.',
        benchmark=benchmark['all_exact'],weights_unchanged=True,training_mechanism_changed=False))
    # Preserve the exact rejected implementation alongside its negative result.
    code=(ROOT/'python/junqi_search_v3/graph_inference.py').read_text()
    code=code.replace('        self._shape_uses = {}\n\n    def _refresh_revision(self):\n        previous=self._revision\n        super()._refresh_revision()\n        if previous!=self._revision: self._shape_uses.clear()\n','')
    begin=code.index('        self._shape_uses[size]=')
    end=code.index('            if size not in self._graphs:',begin)
    code=code[:begin]+'        if self.cuda_graphs and size not in self._graph_failures:\n'+code[end:]
    archive=out/'REJECTED_UNBOUNDED_GRAPH_SOURCE.py';archive.write_bytes(code.encode('utf-8'))
    expected=read_json(out/'BENCHMARK_REJECTED_UNBOUNDED_CAPTURE.json')['protocol']['optimized_sha']
    assert sha(archive)==expected,'negative implementation archive must match original benchmark SHA'
    cases=[r for r in benchmark['cases'] if r['mode']=='evaluation']
    table=''.join(f'<tr><td>{r["roots"]} 个局面，256搜索</td><td>{r["median_speedup"]:.2f}×</td><td>动作、访问、先验、Q与Policy完全一致</td></tr>' for r in cases)
    text=f'''<!doctype html><meta charset="utf-8"><title>性能优化验证与接续</title>
    <style>body{{font:17px/1.7 "Microsoft YaHei",sans-serif;max-width:1080px;margin:45px auto;color:#173042;background:#f5f8fc}}h1{{font-size:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:14px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:14px;border-bottom:1px solid #ddd;text-align:left}}.note{{color:#617385}}</style>
    <h1>性能优化验证与接续</h1><section><b>采用：评测推理提速与可恢复评测。</b><p>训练搜索和学习机制保留原实现；没有提高学习次数，也没有更改学习率、搜索预算、教师比例或标签。</p></section>
    <section><h2>预热后交替对照，3次重复</h2><table><tr><th>相同计算条件</th><th>实测搜索提速</th><th>核验</th></tr>{table}</table><p>BF16、物理batch32，原256搜索；没有额外模拟、结果缓存、树复用或形状合并。5个当前/保护模型逐项一致，模型权重不变。</p><p class="note">这些是保存局面的搜索测量，不能当作整轮训练或完整比赛已加速同样倍数。实际墙钟继续记录。</p></section>
    <section><h2>保留的阴性结果</h2><p>对新批量立即捕获曾慢约8%。限制捕获后，小批量热态能提速，但按实际训练轨迹批量重测（含准备开销），仍只有{collector['median_speedup']:.3f}×，约慢{(1/collector['median_speedup']-1)*100:.1f}%。训练侧候选因此不采用。</p></section>
    <section><h2>评测断点与验证</h2><p>每20轮保存活动棋谱与完整比赛分片。显式恢复时核验布局、模型、教师、预算、源码和文件摘要，重建原生状态；完整统计与晋级门槛不变。3项断点/规则测试、5项搜索测试及5120步规则对照通过。</p><p>旧中止的362完整门禁没有完整原始分片，保留旧记录，在新目录完整重测。先完成362→363的256新局/64更新及完整提交，再评测冻结362；当前学习权重不会被晋级模型替换。</p></section>
    <section><h2>训练范围</h2><p>接续完整362到400：38段，计划9728新训练局和2432更新。达到400复盘。自博弈比例和新方法此时仍只讨论，不自动切换。</p><p class="note">教师50毫秒建议可能随运行负载波动，真实身份、原二进制与重复来源监测持续保留。开发评测不进训练回放，最终布局不使用。</p></section>'''
    (out/'REPORT.html').write_text(text,encoding='utf-8')
    atomic_json(out/'ARTIFACTS.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='ARTIFACTS.json'})
    print(json.dumps(dict(report=str(out/'REPORT.html'),all_passed=True,search_speedups=[c['median_speedup'] for c in cases],training_candidate_rejected=True),ensure_ascii=False))
if __name__=='__main__':main()
