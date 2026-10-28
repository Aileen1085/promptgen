"""Exact metric sharing and final-only scoring over the frozen Bone runtime."""
import copy
import hashlib
from pathlib import Path
from tools.bone_shared_floor_probe_runtime import (
    GridSession as PriorSession, install_v92, install_v10, install_restoration, tag_carrier,
)
from tools.bone_floor_protocol import METRICS


class _DeferredValue:
    def __init__(self, result, name):self.result,self.name=result,name
    def __float__(self):return float(self.result()[self.name])
    def __format__(self, spec):return format(float(self),spec)


def deferred_metrics(compute, names=(*METRICS,'pred_voxels','gt_voxels')):
    result=[]
    def get():
        if not result:result.append(compute())
        return result[0]
    return {name:_DeferredValue(get,name) for name in names}


def materialize_metric(metric):
    for name,value in list(metric.items()):
        if isinstance(value,_DeferredValue):metric[name]=float(value)
    return metric


def validate_metric_record(record, control=False):
    proof=record.get('metric_execution',{})
    events=proof.get('events',[])
    if not events or proof.get('provider_sha256')!=hashlib.sha256((Path(__file__).parent/'bone_fast_metrics.py').read_bytes()).hexdigest():
        raise ValueError('missing or changed exact metric provider proof')
    key=(record['task'].get('source_name',record['task'].get('dataset')),record['task']['case_id'],int(record['task'].get('global_class_id',record['task'].get('class_id'))))
    for result in record['results']:
        candidates=[e for e in events if tuple(e['key'])==key and e['floor']==result['floor']]
        if not candidates or any(abs(candidates[-1]['metrics'][name]-result['row'][name])>1e-12 for name in METRICS):
            raise ValueError('final row is not the actual native-grid metric output')
        if control and any(not e['native_compared'] or e['max_abs_error'] is None or e['max_abs_error']>1e-6 for e in candidates):
            raise ValueError('fresh control lacks exact native metric comparison')
    if proof['stats'].get('comparison_failures',0):raise ValueError('native metric comparison failed')


class GridSession(PriorSession):
    def __init__(self, model, pending, fingerprint, on_class, **kwargs):
        self.metric_trackers=[];self.metric_baselines={};self.metric_events=[]
        self.deferred_candidates=0;self.deferred_materialized=0
        def committed(record):
            snapshots=[t.snapshot() for t in self.metric_trackers]
            stats={}
            for tracker,after in zip(self.metric_trackers,snapshots):
                before=self.metric_baselines[id(tracker)]
                for name,value in after.items():
                    if name.startswith('last_') or name in ('workers','cache_capacity','comparison_atol','cache_entries'):continue
                    if isinstance(value,(int,float)) and not isinstance(value,bool):stats[name]=stats.get(name,0)+value-before.get(name,0)
            record['metric_execution']=dict(provider_sha256=hashlib.sha256((Path(__file__).parent/'bone_fast_metrics.py').read_bytes()).hexdigest(),
                stats=stats,providers=snapshots,events=copy.deepcopy(self.metric_events),
                deferred_candidates=self.deferred_candidates,deferred_materialized=self.deferred_materialized)
            validate_metric_record(record,getattr(self,'shared_control',False))
            return on_class(record)
        super().__init__(model,pending,fingerprint,committed,**kwargs)
        from tools import ct13_full_validation_eval as b
        native_provider=b.metric_module
        def provider():
            from tools.bone_fast_metrics import install_provider
            module=native_provider()
            metric,tracker=install_provider(module,compare_native=lambda:bool(getattr(self,'shared_control',False)),workers=2)
            self.metric_trackers.append(tracker);self.metric_baselines[id(tracker)]=tracker.snapshot()
            def score(pred,target,spacing_dhw=None):
                # Private AutoExpand contact fields remain eager. Native v9's
                # intermediate public metric values are not read by expansion.
                floor=self.floor;key=self.current
                def compute():
                    value=metric(pred,target,spacing_dhw=spacing_dhw)
                    state=tracker.snapshot()
                    self.metric_events.append(dict(key=list(key),floor=floor,metrics={n:float(value[n]) for n in METRICS},
                        input_sha256=state['last_input_sha256'],cache_hit=state['last_cache_hit'],
                        native_compared=state['last_native_compared'],max_abs_error=state['last_comparison_max_abs_error']))
                    if self.model=='v9_2':self.deferred_materialized+=1
                    return value
                if self.model=='v9_2':
                    self.deferred_candidates+=1
                    return deferred_metrics(compute)
                return compute()
            module.binary_prompt_metrics_from_masks=score
            return module
        b.metric_module=provider

    def begin(self, task):
        value=super().begin(task)
        self.metric_events=[];self.deferred_candidates=0;self.deferred_materialized=0
        for tracker in self.metric_trackers:
            tracker.clear();self.metric_baselines[id(tracker)]=tracker.snapshot()
        return value

    def emit(self, task, metric, audit, **geometry):
        materialize_metric(metric)
        return super().emit(task,metric,audit,**geometry)
