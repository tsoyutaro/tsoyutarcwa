"""Portable integrity checks for the frozen validation package.

Accept only UTF-8 BOM and universal-newline differences. Expected hashes must
come from the distributor's trusted manifest; never bless current local files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

POLICY = 'utf8-bom-and-universal-newlines-v1'


def normalized_bytes(data: bytes) -> bytes:
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    return data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')


def text_sha256(path: Path) -> str:
    return hashlib.sha256(normalized_bytes(path.read_bytes())).hexdigest()


def verify_frozen_sources(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root/'frozen_production_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Frozen source manifest must be a nonempty dictionary.')
    mismatches, equivalents = [], []
    exact = 0
    for relative, expected in manifest.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Frozen source path escapes the package: '+relative)
        if expected.get('text_normalization') != POLICY or not expected.get('sha256_text'):
            raise ValueError('Outdated integrity manifest: transfer the updated frozen_production_sources.json.')
        if not path.is_file():
            mismatches.append(dict(file=relative, reason='missing_file'))
            continue
        raw = path.read_bytes()
        actual_raw = hashlib.sha256(raw).hexdigest()
        actual_text = hashlib.sha256(normalized_bytes(raw)).hexdigest()
        if actual_text != expected['sha256_text']:
            mismatches.append(dict(file=relative, reason='content_changed',
                expected_sha256=expected['sha256'], actual_sha256=actual_raw,
                expected_sha256_text=expected['sha256_text'], actual_sha256_text=actual_text))
        elif actual_raw == expected['sha256']:
            exact += 1
        else:
            equivalents.append(relative)
    return dict(status='failed' if mismatches else 'passed', policy=POLICY,
        checked_files=len(manifest), byte_identical_files=exact,
        newline_or_bom_only_files=equivalents, mismatches=mismatches)


def require_frozen_sources(root: Path) -> dict:
    result = verify_frozen_sources(root)
    if result['status'] != 'passed':
        raise RuntimeError('Frozen production source content differs after newline/BOM normalization. '
            'Transfer a clean validation package; do not regenerate expected hashes from these files.\n'
            + json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        result = verify_frozen_sources(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(status='failed', error=str(error), policy=POLICY)
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
