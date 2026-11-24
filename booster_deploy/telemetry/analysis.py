"""Stream analysis uses a disk index, never retains whole-session model tensors."""
from collections import Counter,defaultdict,deque
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import numpy as np
from .config import Config
from .reader import iter_file
from .run_config import read_execution_config

STAGES=('publish','retarget','postprocess','encoder','state','keys','events','dashboard','health')


def _ids(values):
    return [tuple(v) for v in values or [] if isinstance(v,(tuple,list)) and len(v)==2 and isinstance(v[0],str) and type(v[1]) is int and v[1]>0]


def _window_ids(value):
    if isinstance(value,dict):
        if isinstance(value.get('id'),str): yield value['id']
        else:
            for key in ('identity','window'):
                yield from _window_ids(value.get(key))
    elif isinstance(value,(tuple,list)):
        for v in value: yield from _window_ids(v)


def _measured(value):
    if isinstance(value,dict):
        kind = value.get('kind')
        nonlive = kind in ('measured_seed','standing_config','activation_hold','default_seed','measurement','configuration')
        if not nonlive and kind not in (None, 'window', 'blend', 'interpolation'):
            return False
        # Every declared input must be non-live. One seed cannot hide a mixed
        # window's missing live source_ids; arbitrary parameters are not lineage.
        children = [value[key] for key in ('identity','window','origins','parents','positions') if value.get(key)]
        return not value.get('source_ids') and (nonlive or bool(children)) and all(_measured(child) for child in children)
    if isinstance(value,(tuple,list)): return bool(value) and all(_measured(v) for v in value)
    return False


# Snapshot-only evidence; arrays are inspected once and never stored in SQLite.
_NUMERIC = {
    'state': ('q', 'dq', 'root_quat_w', 'root_ang_vel_b'),
    'postprocess': ('joint_pos', 'joint_vel', 'root_pos', 'root_quat'),
    'encoder.entry': ('feeds',),
    'encoder.result': ('raw_action', 'clipped_action', 'target'),
    'encoder.physical_target': ('physical_target',),
}
_POST_KINDS = ('measured_seed', 'standing_config', 'activation_hold', 'blend',
               'sample', 'resample', 'interpolation', 'window')
_CHANNELS = ('raw_body', 'retarget_input')


def _tick_parts(value):
    if not isinstance(value, str) or ':' not in value:
        return None
    session, ordinal = value.rsplit(':', 1)
    if not _known(session) or not ordinal.isdecimal():
        return None
    ordinal = int(ordinal)
    return (session, ordinal) if ordinal > 0 else None


def _known(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in ('none', 'unknown')


def _lineage_descriptor(value):
    """Identity-only metadata, never tensor payloads or build/source fingerprints."""
    if isinstance(value, dict):
        names = ('id', 'kind', 'source_channel', 'origins', 'parents', 'positions',
                 'position', 'identity', 'window', 'source_ids')
        return {k: _lineage_descriptor(v) for k, v in value.items() if k in names}
    if isinstance(value, (list, tuple)):
        return [_lineage_descriptor(v) for v in value]
    return value if value is None or isinstance(value, (str, int, bool)) else None


class _OfflineEvidence:
    def __init__(self, db, config, topology):
        self.db, self.config, self.topology = db, config, topology
        self.fields = defaultdict(Counter)
        self.numeric_scope = {key: list(names) for key, names in _NUMERIC.items() if key != 'postprocess'}
        self.numeric_scope['postprocess'] = []
        self.holds = 0
        db.executescript('''
            CREATE TABLE evidence(id INTEGER PRIMARY KEY, kind TEXT, key TEXT, ok INTEGER, verified INTEGER DEFAULT 0);
            CREATE INDEX evidence_key ON evidence(kind,key);
            CREATE TABLE evidence_edges(child INTEGER, kind TEXT, key TEXT, UNIQUE(child,kind,key));
            CREATE TABLE delivery(session TEXT, ordinal INTEGER, key TEXT, valid INTEGER);
            CREATE TABLE descriptors(node INTEGER PRIMARY KEY, metadata TEXT);
            CREATE TABLE claims(child INTEGER, key TEXT, expected TEXT);
        ''')

    def node(self, kind, key, ok=True):
        key = json.dumps(key, sort_keys=True) if key is not None else None
        return self.db.execute('INSERT INTO evidence(kind,key,ok) VALUES (?,?,?)',
                               (kind, key, int(ok and key is not None))).lastrowid

    def edge(self, child, kind, key):
        self.db.execute('INSERT OR IGNORE INTO evidence_edges VALUES (?,?,?)',
                        (child, kind, json.dumps(key, sort_keys=True) if key is not None else None))

    def reject(self, child):
        self.db.execute('UPDATE evidence SET ok=0 WHERE id=?', (child,))

    def sources(self, child, data, require=True):
        sources = _ids(data.get('source_ids'))
        channel = data.get('source_channel')
        if (require and not sources) or (sources and channel not in _CHANNELS) or any(not _known(s[0]) for s in sources) or len(sources) != len(data.get('source_ids') or []):
            self.reject(child)
        for source in sources:
            self.edge(child, 'publish', [channel, source])
            if self.topology == 'retarget':
                self.edge(child, 'retarget', source)

    def origins(self, child, value, channel=None):
        # A referenced ID must resolve to its own recorded postprocess output.
        if isinstance(value, dict):
            channel = value.get('source_channel', channel)
            if 'source_ids' in value:
                # Explicit wrapper sources are dependencies too, even around seeds.
                self.sources(child, dict(value, source_channel=channel), require=False)
            if value.get('kind') is not None and value['kind'] not in _POST_KINDS + ('measurement','configuration'):
                self.reject(child)
            declared = False
            for field in ('identity', 'window'):
                if value.get(field) is not None:
                    declared = True
                    self.origins(child, value[field], channel)
            if _known(value.get('id')):
                self.edge(child, 'post', value['id'])
                expected = _lineage_descriptor(value)
                expected.pop('id', None)
                if expected:
                    self.db.execute('INSERT INTO claims VALUES (?,?,?)',
                                    (child, json.dumps(value['id']), json.dumps(expected, sort_keys=True)))
                # The reference itself is sufficient, but all explicitly embedded
                # lineage must also be resolved, not hidden behind that ID.
                for field in ('origins', 'parents', 'positions'):
                    for origin in value.get(field) or []:
                        self.origins(child, origin, value.get('source_channel', channel))
                return
            # Non-live labels do not waive explicitly declared parents/positions.
            for field in ('origins', 'parents', 'positions'):
                for origin in value.get(field) or []:
                    declared = True
                    self.origins(child, origin, value.get('source_channel', channel))
            if value.get('kind') in _POST_KINDS:
                # Synthetic/seed references need their own unique recorded owner.
                self.reject(child)
                return
            if declared or value.get('kind') in ('measurement', 'configuration'):
                return
        elif isinstance(value, (list, tuple)) and value:
            if len(_ids([value])) == 1:
                if channel not in _CHANNELS or not _known(value[0]):
                    self.reject(child)
                self.edge(child, 'publish', [channel, tuple(value)])
                if self.topology == 'retarget':
                    self.edge(child, 'retarget', tuple(value))
                return
            for origin in value:
                self.origins(child, origin, channel)
            return
        self.reject(child)

    def numeric(self, stage, data):
        scope = stage + '.' + str(data.get('phase')) if stage == 'encoder' else stage
        # Window identity-only records are not numerical reference-frame outputs.
        if scope == 'postprocess' and (data.get('identity') or {}).get('kind') == 'window':
            return
        names = _NUMERIC.get(scope, ())
        if scope == 'postprocess':
            kind = (data.get('identity') or {}).get('kind')
            if kind not in _POST_KINDS:
                names = ()
                self.fields['postprocess.schema'].update(requested=1, invalid=1)
            for field in names:
                if field not in self.numeric_scope[scope]:
                    self.numeric_scope[scope].append(field)
        for field in names:
            self.numeric_value(stage, scope, [field], data.get(field))

    def numeric_value(self, stage, scope, path, value):
        payload = object()
        probe = payload
        for part in reversed(path):
            probe = {part: probe}
        selected = self.config.select(stage, probe)
        enabled = selected is not None
        for part in path:
            if not isinstance(selected, dict) or part not in selected:
                enabled = False
                break
            selected = selected[part]
        counter = self.fields[scope + '.' + '.'.join(path)]
        counter['requested'] += 1
        if not enabled:
            counter['disabled'] += 1
        elif value is None:
            counter['missing'] += 1
        elif isinstance(value, dict):
            # Enumerate configured names as well: disabled leaves were removed by Writer.
            rule = self.config.values.get('stages', {}).get(stage, {})
            for part in path:
                rule = rule.get('fields', rule).get(part, {}) if isinstance(rule, dict) else {}
            names = set(value)
            if isinstance(rule, dict):
                names.update(k for k in rule.get('fields', rule) if k != 'enabled')
            if not names:
                counter['invalid'] += 1
            else:
                counter['valid'] += 1
                for name in sorted(names):
                    self.numeric_value(stage, scope, path + [name], value.get(name))
        elif isinstance(value, np.ndarray) and value.size and value.dtype.kind in 'iufc' and np.isfinite(value).all():
            counter['valid'] += 1
        else:
            counter['invalid'] += 1

    def record(self, stage, data):
        self.numeric(stage, data)
        if stage == 'publish':
            channel = data.get('channel')
            for source in _ids(data.get('source_ids')):
                self.node('publish', [channel, source], channel in _CHANNELS and _known(source[0]) and self.config.enabled(stage))
        elif stage == 'retarget':
            for source in _ids(data.get('source_ids')):
                child = self.node('retarget', source, self.config.enabled(stage))
                self.edge(child, 'publish', [data.get('source_channel'), source])
        elif stage == 'postprocess':
            identity = data.get('identity') or {}
            child = self.node('post', identity.get('id'), _known(identity.get('id')) and identity.get('kind') in _POST_KINDS and self.config.enabled(stage))
            metadata = _lineage_descriptor(identity)
            if 'source_channel' not in metadata:
                metadata['source_channel'] = data.get('source_channel')
            self.db.execute('INSERT INTO descriptors VALUES (?,?)', (child, json.dumps(metadata, sort_keys=True)))
            measured = _measured(identity)
            self.sources(child, data, require=not measured and not identity.get('origins'))
            if 'source_ids' in identity:
                self.sources(child, dict(identity, source_channel=identity.get('source_channel', data.get('source_channel'))), require=False)
            for field in ('identity', 'window'):
                if identity.get(field) is not None:
                    self.origins(child, identity[field], identity.get('source_channel', data.get('source_channel')))
            if identity.get('kind') in ('window', 'blend', 'interpolation') and not identity.get('origins'):
                self.reject(child)
            for origin in identity.get('origins') or []:
                self.origins(child, origin, identity.get('source_channel', data.get('source_channel')))
            for parent in identity.get('parents') or []:
                self.origins(child, parent, identity.get('source_channel', data.get('source_channel')))
            for position in identity.get('positions') or []:
                self.origins(child, position, identity.get('source_channel', data.get('source_channel')))
        elif stage == 'state':
            self.node('state', data.get('state_id'), _known(data.get('state_id')) and self.config.enabled(stage))
        elif stage == 'encoder':
            tick, phase = data.get('policy_tick'), data.get('phase')
            valid = _tick_parts(tick) is not None and self.config.enabled(stage)
            if phase == 'entry':
                child = self.node('entry', tick, valid)
                self.edge(child, 'state', data.get('state_id'))
                self.sources(child, data, require=not _measured(data.get('window')))
                self.origins(child, data.get('window'), data.get('source_channel'))
            elif phase == 'result':
                child = self.node('result', tick, valid)
                self.edge(child, 'entry', tick)
                session, ordinal = _tick_parts(tick) or (None, None)
                self.db.execute('INSERT INTO delivery VALUES (?,?,?,?)', (session, ordinal, json.dumps(tick), int(valid)))
            elif phase == 'physical_target':
                if tick is None and data.get('target_kind') == 'local_hold_or_transition':
                    self.holds += 1
                else:
                    child = self.node('target', tick, valid)
                    self.edge(child, 'result', tick)

    def finish(self):
        # Partial inline declarations must agree with the recorded owner itself;
        # having two different but individually complete parent chains is not proof.
        for child, key, expected in self.db.execute('SELECT child,key,expected FROM claims'):
            owner = self.db.execute("""SELECT d.metadata FROM evidence n JOIN descriptors d ON d.node=n.id
                WHERE n.kind='post' AND n.key=?""", (key,)).fetchone()
            expected = json.loads(expected)
            actual = json.loads(owner[0]) if owner else {}
            if not owner or any(k not in actual or actual[k] != v for k,v in expected.items()):
                self.reject(child)
        # Only referenced identities must be unique. Repeated target publishes are facts,
        # each independently linked to its own policy invocation, never deduplicated.
        self.db.execute('''UPDATE evidence SET ok=0 WHERE key IS NULL OR (kind != 'target' AND
            (SELECT count(*) FROM evidence other WHERE other.kind=evidence.kind AND other.key=evidence.key) != 1)''')
        while True:
            changed = self.db.execute('''UPDATE evidence SET verified=1 WHERE ok=1 AND verified=0
                AND NOT EXISTS (SELECT 1 FROM evidence_edges e WHERE e.child=evidence.id
                    AND NOT EXISTS (SELECT 1 FROM evidence p WHERE p.kind=e.kind AND p.key=e.key AND p.verified=1))''').rowcount
            if not changed:
                break
        requested, matched = self.db.execute("SELECT count(*),coalesce(sum(verified),0) FROM evidence WHERE kind='target'").fetchone()
        trace = dict(requested=requested, matched=matched, unknown=requested-matched,
                     local_hold_or_transition=self.holds, verified=bool(requested and matched == requested),
                     scope='physical_target records -> own result/entry/state/window/origins/channel/source; no execution ACK')
        gaps = dict(prefix=0, internal=0, suffix=0, all_missing=0, unknown=0)
        results = self.db.execute('SELECT count(*) FROM delivery').fetchone()[0]
        delivered = 0
        self.db.executescript('''
            CREATE INDEX delivery_key ON delivery(key);
            CREATE TABLE delivery_bounds AS SELECT d.session, min(d.ordinal) low, max(d.ordinal) high
                FROM delivery d WHERE d.valid=1 AND EXISTS
                (SELECT 1 FROM evidence t WHERE t.kind='target' AND t.key=d.key AND t.ok=1)
                GROUP BY d.session;
            CREATE UNIQUE INDEX delivery_bounds_session ON delivery_bounds(session);
        ''')
        for session, ordinal, key, valid, low, high in self.db.execute('''
                SELECT d.session,d.ordinal,d.key,d.valid,b.low,b.high FROM delivery d
                LEFT JOIN delivery_bounds b ON b.session=d.session ORDER BY d.session,d.ordinal'''):
            has_target = self.db.execute("SELECT 1 FROM evidence WHERE kind='target' AND key=? AND ok=1 LIMIT 1", (key,)).fetchone()
            if has_target:
                delivered += 1
                continue
            if not valid:
                gaps['unknown'] += 1
                continue
            category = 'all_missing' if low is None else 'prefix' if ordinal < low else 'suffix' if ordinal > high else 'internal'
            gaps[category] += 1
        delivery = dict(requested=results, delivered=delivered, gaps=gaps,
                        verified=bool(results and not sum(gaps.values())), scope='all recorded results, by session and numeric policy tick; no prefix exemption')
        if not self.numeric_scope['postprocess']:
            self.numeric_scope['postprocess'] = list(_NUMERIC['postprocess'])
        for scope, names in self.numeric_scope.items():
            stage = scope.split('.')[0]
            for field in names:
                key = scope + '.' + field
                if key not in self.fields:
                    self.fields[key]['not_observed' if self.config.enabled(stage) else 'disabled'] += 1
        numeric = dict(scope=self.numeric_scope,
                       fields={key:dict(value) for key,value in sorted(self.fields.items())},
                       not_checked=['publish/retarget raw payload schemas', 'postprocess window identity-only shells (no tensor requirement)', 'LowState bytes decoding',
                                    'model expected feed-name/schema equivalence', 'numerical replay and robot execution'])
        numerical_complete = all(c.get('valid', 0) > 0 and not any(c.get(k, 0) for k in ('missing','disabled','invalid','not_observed')) for c in self.fields.values())
        return trace, delivery, numeric, numerical_complete


def analyze(directory):
    directory=Path(directory)
    path=directory/'manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else {'topology':'unknown','config':{}}
    config=Config(manifest.get('config',{}))
    execution_config=read_execution_config(directory)
    metadata=manifest.get('metadata',{})
    topology=execution_config.get('topology') or manifest.get('topology') or metadata.get('topology') or 'unknown'
    counts=Counter(); errors=[]; issues=Counter(); writer_counts=defaultdict(Counter); terminals={}
    if topology not in ('retarget', 'unknown'):
        issues['unsupported_topology'] += 1
    task=execution_config.get('task') or metadata.get('task')
    if task not in (None, 't2_gmc_rt'):
        issues['unsupported_task'] += 1
    events=Counter(); keys=Counter(); last={}; intervals=defaultdict(lambda:deque(maxlen=8192)); dimensions=defaultdict(set)
    with tempfile.TemporaryDirectory(prefix='telemetry-analysis-',dir=directory) as temp:
        db=sqlite3.connect(Path(temp)/'index.sqlite')
        db.executescript('CREATE TABLE ids(stage TEXT, key TEXT, PRIMARY KEY(stage,key)); CREATE TABLE requests(up TEXT,down TEXT,key TEXT,PRIMARY KEY(up,down,key)); CREATE TABLE seq(stream TEXT, ordinal INTEGER, PRIMARY KEY(stream,ordinal));')
        evidence = _OfflineEvidence(db, config, topology)
        def add(stage,key):
            if key is None: issues[stage+'_identity_unknown']+=1; return
            db.execute('INSERT OR IGNORE INTO ids VALUES (?,?)',(stage,json.dumps(key,sort_keys=True)))
        def request(up,down,key):
            if key is None: issues[up+'_to_'+down+'_identity_unknown']+=1; return
            db.execute('INSERT OR IGNORE INTO requests VALUES (?,?,?)',(up,down,json.dumps(key,sort_keys=True)))
        for file in sorted(directory.glob('*.msgpack')):
            for row in iter_file(file,errors):
                stage=row['stage']; data=row.get('data',{}); token=row.get('writer_token','unknown')
                if stage=='terminal': terminals[token]=data; continue
                counts[stage]+=1; writer_counts[token][stage]+=1
                evidence.record(stage, data)
                stamp=row.get('t_mono_ns'); gapkey=(token,stage,data.get('phase'),data.get('channel'),data.get('component'),str(data.get('source_id',['unknown'])[0]))
                if isinstance(stamp,int):
                    if gapkey in last: intervals[str(gapkey)].append((stamp-last[gapkey])/1e6)
                    last[gapkey]=stamp
                    if stamp<row.get('origin_mono_ns',stamp): issues['time_before_origin']+=1
                sources=_ids(data.get('source_ids'))
                if stage in ('retarget','postprocess'):
                    for source in sources: add(stage,source)
                if stage=='publish':
                    if not sources: issues['publish_source_unknown']+=1
                    channel=data.get('channel')
                    if channel in _CHANNELS:
                        for source in sources: add('publish',[channel,source])
                    else: issues['publish_channel_unknown']+=1
                    for stream,ordinal in sources:
                        db.execute('INSERT OR IGNORE INTO seq VALUES (?,?)',(stream,ordinal))
                if stage=='retarget':
                    if not sources: issues['retarget_source_unknown']+=1
                    for source in sources: request('publish','retarget',[data.get('source_channel','unknown'),source])
                if stage=='postprocess':
                    identity=data.get('identity') or {}
                    if identity.get('id'): add('post_identity',identity['id'])
                    else: issues['synthetic_identity_unknown']+=1
                    if identity.get('kind') in ('interpolation','blend') and identity.get('weight') is None:
                        issues['synthetic_weight_unknown']+=1
                    if topology=='retarget':
                        for source in sources: request('retarget','postprocess',source)
                if stage=='state':
                    add('state',data.get('state_id'))
                    if data.get('state_source')=='real' and data.get('low_state') is None: issues['real_low_state_unknown']+=1
                if stage=='encoder':
                    tick=data.get('policy_tick'); phase=data.get('phase')
                    if phase=='entry':
                        add('entry',tick)
                        request('state','encoder',data.get('state_id'))
                        for identity in _window_ids(data.get('window')): request('post_identity','encoder',identity)
                        for source in sources:
                            request('publish','encoder',[data.get('source_channel','unknown'),source])
                            if topology=='retarget': request('retarget','encoder',source)
                        if not sources and not _measured(data.get('window')): issues['encoder_source_unknown']+=1
                    elif phase=='result':
                        add('result',tick); request('entry','result',tick)
                    elif phase=='physical_target':
                        if tick is not None: add('physical_target',tick); request('result','physical_target',tick)
                        elif data.get('target_kind') not in ('local_hold_or_transition',): issues['target_policy_tick_unknown']+=1
                    for field in ('feeds','raw_action','clipped_action','target','physical_target'):
                        value=data.get(field)
                        values=value.values() if isinstance(value,dict) else [value]
                        for name,array in enumerate(values):
                            if isinstance(array,np.ndarray):
                                dimensions[field].add(tuple(array.shape))
                                if array.dtype.kind in 'fc' and not np.isfinite(array).all(): issues[field+'_nonfinite']+=1
                                if field=='feeds' and (array.ndim<2 or array.shape[0]!=1): issues['model_input_dimension']+=1
                    raw=data.get('raw_action'); clipped=data.get('clipped_action'); target=data.get('target')
                    if isinstance(raw,np.ndarray) and isinstance(clipped,np.ndarray) and raw.size!=clipped.size: issues['action_dimension_mismatch']+=1
                    if isinstance(target,np.ndarray) and isinstance(clipped,np.ndarray) and target.size!=clipped.size: issues['target_dimension_mismatch']+=1
                if stage=='events': events[data.get('event','unknown')]+=1
                if stage=='keys': keys[str(data.get('key'))+':'+str(data.get('phase'))]+=1
        db.commit()
        links={}
        for up,down,n in db.execute('SELECT up,down,count(*) FROM requests GROUP BY up,down'):
            matched=db.execute('SELECT count(*) FROM requests r JOIN ids i ON i.stage=r.up AND i.key=r.key WHERE r.up=? AND r.down=?',(up,down)).fetchone()[0]
            enabled=config.enabled(up if up in STAGES else 'postprocess' if up=='post_identity' else 'encoder')
            links[up+'_to_'+down]=dict(requested=n,matched=matched,unknown=n-matched,status='checked' if enabled else 'disabled')
        coverage={}
        coverage['missing_identity_stages']=[stage for stage in ('entry','state','result','physical_target')
            if not db.execute('SELECT 1 FROM ids WHERE stage=? LIMIT 1',(stage,)).fetchone()]
        for left,right,name in [('entry','result','missing_result'),('result','physical_target','result_without_local_target')]:
            coverage[name]=db.execute('SELECT count(*) FROM ids l LEFT JOIN ids r ON r.stage=? AND r.key=l.key WHERE l.stage=? AND r.key IS NULL',(right,left)).fetchone()[0]
        continuity=[]
        for stream,n,low,high in db.execute('SELECT stream,count(*),min(ordinal),max(ordinal) FROM seq GROUP BY stream'):
            continuity.append(dict(stream=stream,observed_unique=n,first=low,last=high,unknown_ordinals=high-low+1-n,
                                   prefix_unknown=low!=1,interpretation='Unrecorded or unpublished ordinals have unknown causes; they do not prove network packet loss'))
        trace, delivery, numeric, numerical_complete = evidence.finish()
        db.close()
    writer_results=[]
    registered=set()
    for marker in directory.glob('writer.*.json'):
        try: registered.add(json.loads(marker.read_text())['writer_token'])
        except (OSError,ValueError,KeyError) as exc: errors.append(dict(file=str(marker),error=repr(exc)))
    for token in set(writer_counts)|set(terminals)|registered:
        terminal=terminals.get(token,{})
        observed=dict(writer_counts[token])
        complete=terminal.get('state')=='closed' and not terminal.get('incomplete',True) and terminal.get('rejected',0)==0 and not terminal.get('error')
        complete=complete and terminal.get('stage_counts')==observed and terminal.get('accepted')==sum(observed.values())
        writer_results.append(dict(token=token,complete=bool(complete),terminal=terminal,observed=observed))
    required={'publish','retarget','postprocess','encoder','state','events','health'} if topology=='retarget' else set()
    stages={stage:('disabled' if not config.enabled(stage) else 'observed' if counts[stage] else 'unavailable' if stage in required else 'not_observed_optional') for stage in STAGES}
    metrics={key:dict(samples=len(v),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),maximum=max(v)) for key,v in intervals.items() if v}
    complete=bool(writer_results) and all(w['complete'] for w in writer_results) and not errors
    coverage['missing_expected_stages']=[stage for stage,value in stages.items() if value=='unavailable']
    coverage['disabled_expected_stages']=sorted(stage for stage in required if stages[stage]=='disabled')
    coverage_verified=complete and topology=='retarget' and not coverage['missing_expected_stages'] and not coverage['disabled_expected_stages'] and not coverage['missing_identity_stages'] and not coverage['missing_result'] and not coverage['result_without_local_target'] and not issues and all(v['unknown']==0 and v['status']=='checked' for v in links.values())
    return dict(schema_version=4,run_id=manifest.get('run_id',directory.name),topology=topology,complete=complete,file_integrity_complete=complete,identity_coverage_verified=coverage_verified,coverage_verified=coverage_verified,coverage=coverage,
        issued_target_traceability=dict(trace, verified=complete and trace["verified"]),
        result_delivery=delivery, numerical_fields=numeric, numerical_fields_complete=complete and numerical_complete,
        stages=stages,stage_counts=dict(counts),links=links,source_continuity=continuity,issues=dict(issues),reader_errors=errors,
        writers=writer_results,interval_ms=metrics,dimensions={k:[list(s) for s in sorted(v)] for k,v in dimensions.items()},
        key_events=dict(keys),lifecycle_events=dict(events),execution_ack=False,
        limitations=['coverage_verified / identity_coverage_verified cover stage and ID associations only, not core numerical-field completeness or numerical model reproduction','Publishing a target/low_cmd is not an execution ACK','Retarget frame selection is a consumption policy, not source packet loss','Interval percentiles retain at most the latest 8192 samples per stream','Hop-by-hop IDs are indexed in temporary SQLite; disabled or unknown sources cannot establish completeness','Missing stages are not inferred when the topology is unknown'])


def markdown(report, language='zh'):
    if language not in ('en', 'zh', 'zh-CN'):
        raise ValueError('Report language must be en, zh, or zh-CN')
    english = language == 'en'
    if english:
        lines=['# Teleoperation telemetry review',f"Run: {report['run_id']}",f"Topology: {report['topology']}",
            f"File lifecycle complete: {'yes' if report['complete'] else 'no/unknown'}",'','## Stages','|Stage|Fact records|Status|','|---|---:|---|']
    else:
        lines=['# 遥操作 telemetry 复盘',f"运行：{report['run_id']}",f"拓扑：{report['topology']}",
            f"文件生命周期完整：{'是' if report['complete'] else '否/未知'}",'','## 环节','|环节|事实记录数|状态|','|---|---:|---|']
    lines += [f"|{stage}|{report['stage_counts'].get(stage,0)}|{status}|" for stage,status in report['stages'].items()]
    lines += ['','## Hop-by-hop associations' if english else '## 逐跳关联',
              '|Hop|Requested|Matched|Unknown|' if english else '|跳|请求|匹配|未知|','|---|---:|---:|---:|']
    lines += [f"|{key}|{value['requested']}|{value['matched']}|{value['unknown']}|" for key,value in report['links'].items()]
    lines += ['', '## Independent offline dimensions (do not replace legacy coverage)' if english else '## 独立离线分项（不替代 legacy coverage）',
              '### issued_target_traceability', json.dumps(report['issued_target_traceability'], ensure_ascii=False),
              '### result_delivery', json.dumps(report['result_delivery'], ensure_ascii=False),
              '### numerical_fields_complete', str(report['numerical_fields_complete']),
              json.dumps(report['numerical_fields'], ensure_ascii=False)]
    lines += ['','## Preliminary anomaly facts' if english else '## 初步异常事实',json.dumps(report['issues'],ensure_ascii=False),
              '','## Keys and lifecycle' if english else '## 按键与生命周期',json.dumps(report['key_events'],ensure_ascii=False),
              json.dumps(report['lifecycle_events'],ensure_ascii=False),'','## Limitations' if english else '## 边界']
    translations = {
        'coverage_verified / identity_coverage_verified cover stage and ID associations only, not core numerical-field completeness or numerical model reproduction': 'coverage_verified / identity_coverage_verified 仅为环节与ID关联覆盖，不验证核心数值字段齐全或模型数值复现',
        'Publishing a target/low_cmd is not an execution ACK': '目标/low_cmd 发布不是执行 ACK',
        'Retarget frame selection is a consumption policy, not source packet loss': 'retarget 择帧属于消费策略，不计源丢包',
        'Interval percentiles retain at most the latest 8192 samples per stream': '间隔分位数最多保留每流最近8192样本',
        'Hop-by-hop IDs are indexed in temporary SQLite; disabled or unknown sources cannot establish completeness': '逐跳ID索引在临时SQLite；禁用/未知来源不可补证完整',
        'Missing stages are not inferred when the topology is unknown': '拓扑未知时不猜测缺失环节',
    }
    lines += ['- '+(v if english else translations.get(v,v)) for v in report['limitations']]
    return '\n'.join(lines)+'\n'


def save(directory):
    directory=Path(directory); report=analyze(directory)
    (directory/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (directory/'analysis.zh-CN.md').write_text(markdown(report))
    (directory/'analysis.en.md').write_text(markdown(report, language='en'))
    return report
