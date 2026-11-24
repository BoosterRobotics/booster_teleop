"""Hierarchical filtering happens before tensor snapshots."""
from pathlib import Path
import yaml

DEFAULT = Path(__file__).with_name('telemetry.yaml')
STAGES=frozenset(('publish','retarget','postprocess','encoder','state','keys','events','dashboard','health'))
#: Small per-second facts the live dashboard tails.  They stay on when
#: persistent recording is disabled; otherwise the pipeline rate bars go blank.
UI_STAGES=frozenset(('keys','events','dashboard','health'))

class Config:
    def __init__(self, values=None):
        self.values = values if values is not None else yaml.safe_load(DEFAULT.read_text())
        self.recording_enabled=bool(self.values.get('enabled',True))
        self.trajectory_switch=self.values.get('trajectory_switch','once')
        if self.trajectory_switch not in ('once','unlimited'):
            raise ValueError('trajectory_switch must be once or unlimited')
        self.recording_mode=self.values.get('recording',{}).get('mode','full_run')
        if self.recording_mode not in ('full_run','tracking'):
            raise ValueError('recording.mode must be full_run or tracking')
        unknown=set(self.values.get('stages',{}))-STAGES
        if unknown: raise ValueError('unknown telemetry stages: '+str(sorted(unknown)))

    @classmethod
    def load(cls, path=None):
        return cls(yaml.safe_load(Path(path or DEFAULT).read_text()) or {})

    def enabled(self, stage):
        node = self.values.get('stages', {}).get(stage, {})
        if node is False or isinstance(node, dict) and not node.get('enabled', True):
            return False
        if stage in UI_STAGES:
            return True
        return bool(self.values.get('enabled', True))

    def field_enabled(self, stage, name):
        if not self.enabled(stage):
            return False
        node = self.values.get('stages', {}).get(stage, {})
        if not isinstance(node, dict):
            return True
        return (node.get('fields', {}) or {}).get(name, True) is not False

    def select(self, stage, payload):
        if not self.enabled(stage):
            return None
        node = self.values.get('stages', {}).get(stage, {})
        return self._filter(payload, node.get('fields', {}) if isinstance(node, dict) else {})

    def _filter(self, value, rule):
        if rule is False or isinstance(rule, dict) and not rule.get('enabled', True):
            return None
        if isinstance(value, dict):
            fields = rule.get('fields', rule) if isinstance(rule, dict) else {}
            return {k: self._filter(v, fields.get(k, {})) for k,v in value.items()
                    if fields.get(k) is not False and not (isinstance(fields.get(k),dict) and not fields[k].get('enabled',True))}
        return value
