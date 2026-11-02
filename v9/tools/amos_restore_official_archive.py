"""Restore only manifest-selected original AMOS files from the official ZIP.

No downloads, dataset/cache regeneration or archive-wide extraction occurs here.
Failed unique temporary files remain as evidence; retries use a fresh temporary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SIZE = 24234336519
OFFICIAL_MD5 = '67717b2a483ac0744c89c3016b7aaef7'
RESERVE_BYTES = 3 * 1024 ** 3
CHUNK = 1024 ** 2


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def manifest_relative(value, role, case_id):
    parts = PurePosixPath(str(value).replace('\\', '/')).parts
    if 'amos22' in parts:
        parts = parts[parts.index('amos22') + 1:]
    if (not 2 <= len(parts) <= 3 or any(p in ('.', '..') or ':' in p for p in parts)
            or any(not re.fullmatch(('images' if role == 'image' else 'labels') + '(Tr|Va|Ts)', p)
                   for p in parts[:-1]) or len(set(parts[:-1])) != 1
            or parts[-1] != case_id + '.nii.gz'):
        raise ValueError('unsafe or inconsistent manifest target: ' + str(value))
    return '/'.join(parts)


def safe_target(root, relative):
    target = root / relative
    current = root
    for component in (None, *PurePosixPath(relative).parts):
        current = current if component is None else current / component
        if current.is_symlink() or getattr(current, 'is_junction', lambda: False)():
            raise ValueError('unsafe target symlink: ' + str(current))
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('target escapes target root: ' + str(target))
    if target.exists() and not target.is_file():
        raise ValueError('existing target is not a regular file: ' + str(target))
    return target


def select_members(archive, manifest):
    members = {}
    for entry in archive.infolist():
        name = entry.orig_filename
        if (name.startswith('/') or '\\' in name or '\x00' in name
                or any(p == '..' or ':' in p for p in name.split('/'))):
            raise ValueError('unsafe ZIP member: ' + name)
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError('ZIP symlink refused: ' + name)
        parts = PurePosixPath(name).parts
        if entry.is_dir() or '__MACOSX' in parts or not re.fullmatch(r'amos_\d{4}\.nii\.gz', parts[-1]):
            continue
        roles = {role for role, prefix in (('image', 'images'), ('label', 'labels'))
                 if any(re.fullmatch(prefix + '(Tr|Va|Ts)', p) for p in parts[:-1])}
        for role in roles:
            members.setdefault((parts[-1][:-7], role), []).append(entry)
    selected = []
    seen = set()
    datasets = manifest['datasets']['amos']
    for split in ('train', 'val'):
        for row in datasets[split]:
            case_id = row['case_id']
            if not re.fullmatch(r'amos_\d{4}', case_id) or case_id in seen:
                raise ValueError('invalid or duplicate manifest case: ' + str(case_id))
            seen.add(case_id)
            for role in ('image', 'label'):
                relative = manifest_relative(row[role], role, case_id)
                matches = members.get((case_id, role), [])
                if len(matches) != 1:
                    raise ValueError('missing or ambiguous archive case/role: ' + case_id + '/' + role)
                selected.append((dict(split=split, case_id=case_id, role=role,
                                      target_relative=relative), matches[0]))
    if not selected:
        raise ValueError('empty AMOS manifest')
    return selected


def historical_metadata(protocol):
    result = {}
    if protocol is None:
        return result
    data = read_json(protocol)['data']
    for name, sha in data['dataset'].items():
        parts = PurePosixPath(name.replace('\\', '/')).parts
        if 'amos22' not in parts or not re.fullmatch(r'amos_\d{4}\.nii\.gz', parts[-1]):
            continue
        role = 'image' if any(p.startswith('images') for p in parts) else 'label'
        relative = manifest_relative(name, role, parts[-1][:-7])
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('invalid historical SHA256: ' + name)
        value = dict(sha256=sha, **data.get('stats', {}).get(name, {}))
        if relative in result and result[relative] != value:
            raise ValueError('conflicting historical metadata: ' + relative)
        result[relative] = value
    return result


def disk_free(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def require_space(path, additional_bytes):
    if disk_free(path) < additional_bytes + RESERVE_BYTES:
        raise ValueError('insufficient disk space to retain 3 GiB reserve')


def process_archive(archive, manifest, *, target_root=None, historical_protocol=None, restore=False, report=None):
    """Inspect without writes by default; restoration requires official size/MD5."""
    archive, manifest = project_path(archive), project_path(manifest)
    root = project_path(target_root) if target_root is not None else ROOT.parent.parent / 'amos22'
    root = root.absolute()
    historical_protocol = project_path(historical_protocol) if historical_protocol is not None else None
    if archive.stat().st_size != OFFICIAL_SIZE:
        raise ValueError('official archive size mismatch')
    md5 = digest(archive, 'md5')
    if md5 != OFFICIAL_MD5:
        raise ValueError('official archive MD5 mismatch')
    audit = {} if report is None else report
    audit.update(mode='restore' if restore else 'inspect',
                 archive=dict(path=str(archive), size=archive.stat().st_size, md5=md5,
                              official_md5_verified=True),
                 manifest_sha256=digest(manifest), target_root=str(root), files=[])
    history = historical_metadata(historical_protocol)
    audit['historical_protocol_sha256'] = digest(historical_protocol) if historical_protocol else None
    with zipfile.ZipFile(archive) as z:
        selected = select_members(z, read_json(manifest))
        for row, entry in selected:
            safe_target(root, row['target_relative'])
            reference = history.get(row['target_relative'], {})
            if 'size' in reference and reference['size'] != entry.file_size:
                raise ValueError('historical size mismatch: ' + row['target_relative'])
            row.update(archive_member=entry.filename, size=entry.file_size,
                       archive_crc32=f'{entry.CRC:08x}', zip_crc_verified=False, sha256=None,
                       historical_sha256=reference.get('sha256'), historical_sha256_verified=False,
                       historical_mtime_restored=False, integrity_provenance=['official_md5'], status='planned')
            audit['files'].append(row)
        audit['selected_uncompressed_bytes'] = sum(e.file_size for _, e in selected)
        required = sum(e.file_size for row, e in selected if not (root / row['target_relative']).exists())
        free = disk_free(root)
        audit.update(required_new_bytes=required, disk_free_bytes=free, reserve_bytes=RESERVE_BYTES,
                     space_sufficient=free >= required + RESERVE_BYTES)
        if not restore:
            return audit
        if not audit['space_sufficient']:
            raise ValueError('insufficient disk space for selected files and 3 GiB reserve')
        for row, entry in selected:
            target = safe_target(root, row['target_relative'])
            exists = target.exists()
            temporary = None
            if not exists:
                require_space(root, entry.file_size)
                target.parent.mkdir(parents=True, exist_ok=True)
                safe_target(root, row['target_relative'])
                temporary = tempfile.NamedTemporaryFile(mode='wb', prefix='.' + target.name + '.restore-',
                                                        suffix='.part', dir=target.parent, delete=False)
            sha = hashlib.sha256()
            try:
                with z.open(entry) as source:
                    for block in iter(lambda: source.read(CHUNK), b''):
                        sha.update(block)
                        if temporary is not None:
                            require_space(target.parent, len(block))
                            temporary.write(block)
                if temporary is not None:
                    temporary.flush()
                    os.fsync(temporary.fileno())
            finally:
                if temporary is not None:
                    temporary.close()
            row.update(sha256=sha.hexdigest(), zip_crc_verified=True)
            checked = target if exists else Path(temporary.name)
            if checked.stat().st_size != entry.file_size or digest(checked) != row['sha256']:
                raise ValueError('existing target differs' if exists else 'written target integrity mismatch')
            reference = history.get(row['target_relative'], {})
            if reference and row['sha256'] != reference['sha256']:
                raise ValueError('historical SHA256 mismatch: ' + str(target))
            row['historical_sha256_verified'] = bool(reference)
            row['integrity_provenance'] += ['zip_crc', 'written_file_sha256']
            if reference:
                row['integrity_provenance'].append('historical_sha256')
            if reference.get('size') == entry.file_size and isinstance(reference.get('mtime_ns'), int):
                os.utime(checked, ns=(checked.stat().st_atime_ns, reference['mtime_ns']))
                if checked.stat().st_mtime_ns != reference['mtime_ns']:
                    raise ValueError('historical mtime could not be restored exactly')
                row['historical_mtime_restored'] = True
            if not exists:
                safe_target(root, row['target_relative'])
                if os.name == 'nt':
                    os.rename(temporary.name, target)  # Windows rename refuses an existing target.
                else:
                    os.link(temporary.name, target)  # POSIX atomic, exclusive publication.
                    Path(temporary.name).unlink()  # Only this invocation's explicit temporary.
            row['status'] = 'reused' if exists else 'restored'
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'restore'))
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--target-root', help='default: project root ../../amos22')
    parser.add_argument('--historical-protocol', help='optional S21 data.dataset/data.stats JSON')
    parser.add_argument('--audit', help='new JSON audit outside target root; existing files are refused')
    args = parser.parse_args(argv)
    report, audit_path, audit_ready, code = {}, None, False, 0
    try:
        root = project_path(args.target_root) if args.target_root else ROOT.parent.parent / 'amos22'
        if args.audit:
            audit_path = project_path(args.audit)
            if audit_path.exists() or audit_path.is_symlink():
                raise ValueError('existing audit path refused: ' + str(audit_path))
            if audit_path.resolve().is_relative_to(root.resolve()):
                raise ValueError('audit must be outside the dataset target root')
        audit_ready = True
        process_archive(args.archive, args.manifest, target_root=args.target_root,
                        historical_protocol=args.historical_protocol,
                        restore=args.command == 'restore', report=report)
        report['status'] = 'complete' if args.command == 'restore' else 'inspected'
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        report.update(status='failed', error=str(exc))
        print(str(exc), file=sys.stderr)
        code = 2
    if audit_path is not None and audit_ready:
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        except OSError as exc:
            print('audit could not be written: ' + str(exc), file=sys.stderr)
            code = 2
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
