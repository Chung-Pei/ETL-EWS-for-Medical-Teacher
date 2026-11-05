# -*- coding: utf-8 -*-
"""Round 7 file-access audit hook.  Import this module FIRST in every script.
Logs every open()/listdir()/scandir() of a path under the working tree or the uploads directory to
logs/r7_file_access_audit.jsonl and hard-blocks any path that contains '1142' (case-insensitive)."""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(_HERE), 'logs', os.environ.get('R7_AUDIT_NAME', 'r7_file_access_audit.jsonl'))
os.makedirs(os.path.dirname(LOG), exist_ok=True)
_FD = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
_WORK = os.path.dirname(_HERE)
_STAG = os.environ.get('R7_STAGING', os.path.join(os.path.dirname(_WORK), 'round7_pack_staging'))
_ROOTS = tuple(os.path.abspath(p) + os.sep for p in (_WORK, _STAG))
_SEEN = set()
_SCRIPT = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else '<interactive>'


def _to_str(p):
    if isinstance(p, int):
        return None
    try:
        return os.fsdecode(p)
    except Exception:
        return None


def _hook(event, args):
    if event not in ('open', 'os.listdir', 'os.scandir', 'glob.glob'):
        return
    p = _to_str(args[0]) if args else None
    if p is None:
        return
    if '1142' in p.lower():
        os.write(_FD, (json.dumps({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'script': _SCRIPT, 'event': event,
                                   'path': p, 'verdict': 'BLOCKED'}) + '\n').encode())
        raise PermissionError(f'Round 7 rule: any path containing 1142 is forbidden: {p}')
    ap = os.path.abspath(p)
    if not ap.startswith(_ROOTS) or ap == LOG:
        return
    mode = args[1] if event == 'open' and len(args) > 1 else ''
    kind = 'write' if (isinstance(mode, str) and any(c in mode for c in 'wax+')) else 'read'
    key = (event, ap, kind)
    if key in _SEEN:
        return
    _SEEN.add(key)
    os.write(_FD, (json.dumps({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'script': _SCRIPT, 'event': event,
                               'path': ap, 'kind': kind, 'verdict': 'allowed'}) + '\n').encode())


sys.addaudithook(_hook)
