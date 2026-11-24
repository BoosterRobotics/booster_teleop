"""Read complete records, preserving valid prefixes and explicit integrity errors."""
from collections import Counter
import json
from pathlib import Path
import msgpack
from .codec import decode

def iter_file(path,errors=None):
    unpacker=msgpack.Unpacker(raw=False,object_hook=decode,max_buffer_size=64*1024*1024)
    last=0
    try:
        with open(path,'rb') as handle:
            while True:
                chunk=handle.read(65536)
                if not chunk: break
                unpacker.feed(chunk)
                for record in unpacker:
                    if not isinstance(record,dict) or not isinstance(record.get('stage'),str) or not isinstance(record.get('data'),dict): raise ValueError('invalid record schema')
                    if record.get('schema_version')==4 and not isinstance(record.get('writer_token'),str): raise ValueError('missing writer token')
                    last=unpacker.tell()
                    yield record
        if last!=Path(path).stat().st_size and errors is not None:
            errors.append({'file':str(path),'error':'truncated','complete_bytes':last})
    except (ValueError,TypeError,KeyError,OverflowError,msgpack.UnpackException,OSError) as exc:
        if errors is not None: errors.append({'file':str(path),'error':repr(exc),'complete_bytes':last})

def read_records(directory):
    rows=[]; errors=[]
    for path in sorted(Path(directory).glob('*.msgpack')): rows.extend(iter_file(path,errors))
    counts={}; terminals={}
    for r in rows:
        token=r.get('writer_token')
        if r['stage']=='terminal': terminals[token]=r.get('data',{})
        else: counts.setdefault(token,Counter())[r['stage']]+=1
    complete=bool(rows) and not errors
    registered=set()
    for marker in Path(directory).glob('writer.*.json'):
        try: registered.add(json.loads(marker.read_text())['writer_token'])
        except (OSError,ValueError,KeyError): complete=False
    for token in set(counts)|set(terminals)|registered:
        terminal=terminals.get(token,{})
        complete=complete and terminal.get('state')=='closed' and not terminal.get('incomplete',True)
        complete=complete and not terminal.get('error') and terminal.get('rejected',0)==0
        complete=complete and terminal.get('accepted')==sum(counts.get(token,{}).values())
        complete=complete and dict(counts.get(token,{}))==terminal.get('stage_counts',{})
    return rows,bool(complete)
