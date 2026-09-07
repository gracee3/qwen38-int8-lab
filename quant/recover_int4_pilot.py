"""Revalidate a fully exported pilot after failure; never repeat GPTQ or alter weights."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import os
from pathlib import Path

from validation.validate_int4 import validate


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'staging', 'output', 'audit', 'corpus', 'failed_report', 'report'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Recovery refuses existing output/report')
    if args.staging.parent != args.output.parent or not args.staging.name.startswith('.' + args.output.name + '.incomplete-'):
        raise ValueError('Expected sibling incomplete staging directory')
    result = json.loads(args.failed_report.read_text())
    audit = json.loads(args.audit.read_text())
    corpus = json.loads((args.corpus / 'manifest.json').read_text())
    if result['status'] != 'failed' or result['profile'] != 'real_source_short_and_long_pilot':
        raise ValueError('Expected failed pilot metadata')
    if result['corpus_sha256'] != corpus['corpus_sha256'] or digest(args.corpus / 'calibration.parquet') != corpus['corpus_sha256']:
        raise ValueError('Corpus identity mismatch')
    if result['source_revision'] != audit['source']['revision'] or result['peaks']['safety_trigger'] or result['peaks']['samples'] <= 0:
        raise ValueError('Source identity or resource gate failed')
    for filename, expected in audit['source']['shard_sha256'].items():
        if digest(args.source / filename) != expected:
            raise ValueError('Source changed: ' + filename)
    result['integrity'] = validate(args.source, args.staging, audit)
    result['recovery'] = {
        'failed_report_sha256': digest(args.failed_report),
        'validator_sha256': digest(Path(__file__).resolve().parents[1] / 'validation/validate_int4.py'),
        'staging': str(args.staging),
        'method': 'revalidate_saved_export_no_requantization',
        'weights_modified': False,
        'runtime_validated': False,
    }
    result['status'] = 'passed'
    marker = args.staging / 'EXPERIMENTAL_NON_PRODUCTION.json'
    with marker.open('x') as stream:
        json.dump({'profile': result['profile'], 'production_authorized': False}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    args.staging.rename(args.output)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'report': str(args.report), 'norm_roundtrips': len(result['integrity']['offset_norm_roundtrip'])}))


if __name__ == '__main__':
    main()
