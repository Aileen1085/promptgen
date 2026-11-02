"""Synthetic AMOS recovery checks; small failed-attempt artifacts are retained."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

try:
    recovery = importlib.import_module('tools.amos_restore_official_archive')
except ImportError:
    recovery = None


class ArchiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(recovery, 'implement the isolated AMOS archive recovery helper')
        self.base = Path(tempfile.mkdtemp(prefix='amos-restore-test-'))
        self.archive = self.base / 'amos22.zip'
        self.target = self.base / 'amos22'
        self.manifest = self.base / 'split.json'
        self.image = b'CT volume bytes'
        self.label = b'GT volume bytes'
        self.row = dict(case_id='amos_0008',
                        image='/old/source/amos22/imagesVa/imagesVa/amos_0008.nii.gz',
                        label='/old/source/amos22/labelsVa/amos_0008.nii.gz', classes=[118])
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(dict(datasets=dict(amos=dict(train=[], val=[self.row])))))

    def make_archive(self, extra=(), image=None):
        with zipfile.ZipFile(self.archive, 'w', compression=zipfile.ZIP_STORED) as z:
            z.writestr('amos22/imagesVa/amos_0008.nii.gz', self.image if image is None else image)
            z.writestr('amos22/labelsVa/amos_0008.nii.gz', self.label)
            for name, payload in extra:
                info = zipfile.ZipInfo(name) if isinstance(name, str) else name
                if isinstance(name, str):
                    info.filename = name  # Preserve raw malicious names on Windows as well.
                z.writestr(info, payload)

    def run_archive(self, restore=False, **kwargs):
        with patch.object(recovery, 'OFFICIAL_SIZE', self.archive.stat().st_size), \
             patch.object(recovery, 'OFFICIAL_MD5', hashlib.md5(self.archive.read_bytes()).hexdigest()):
            return recovery.process_archive(self.archive, self.manifest, target_root=self.target,
                                            restore=restore, **kwargs)

    def test_inspect_selects_roles_preserves_double_directory_and_writes_nothing(self):
        self.make_archive(extra=[('amos22/imagesTs/amos_0500.nii.gz', b'unrelated'),
                                 ('__MACOSX/amos22/imagesVa/._amos_0008.nii.gz', b'metadata')])
        audit = self.run_archive()
        self.assertEqual({f['target_relative'] for f in audit['files']},
                         {'imagesVa/imagesVa/amos_0008.nii.gz', 'labelsVa/amos_0008.nii.gz'})
        self.assertEqual(audit['selected_uncompressed_bytes'], len(self.image) + len(self.label))
        self.assertFalse(self.target.exists())
        self.assertTrue(audit['archive']['official_md5_verified'])

    def test_bad_whole_archive_md5_and_size_block_all_target_writes(self):
        self.make_archive()
        with patch.object(recovery, 'OFFICIAL_SIZE', self.archive.stat().st_size), \
             patch.object(recovery, 'OFFICIAL_MD5', '0' * 32):
            with self.assertRaisesRegex(ValueError, 'MD5'):
                recovery.process_archive(self.archive, self.manifest, target_root=self.target, restore=True)
        with patch.object(recovery, 'OFFICIAL_SIZE', 1):
            with self.assertRaisesRegex(ValueError, 'size'):
                recovery.process_archive(self.archive, self.manifest, target_root=self.target, restore=True)
        self.assertFalse(self.target.exists())

    def test_duplicate_case_role_across_splits_is_refused(self):
        self.make_archive(extra=[('amos22/imagesTr/amos_0008.nii.gz', self.image)])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.run_archive(restore=True)
        self.assertFalse(self.target.exists())

    def test_archive_traversal_and_symlink_are_refused_even_when_unselected(self):
        for bad in ('../escape', '/absolute', 'C:/absolute', 'x/../escape', 'x\\escape'):
            with self.subTest(bad=bad):
                self.make_archive(extra=[(bad, b'bad')])
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    self.run_archive(restore=True)
        info = zipfile.ZipInfo('unselected-link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.make_archive(extra=[(info, b'outside')])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_archive(restore=True)
        self.assertFalse(self.target.exists())

    def test_manifest_escape_or_case_mismatch_is_refused(self):
        self.make_archive()
        for image in ('/old/amos22/../outside/amos_0008.nii.gz',
                      '/old/amos22/imagesVa/amos_0009.nii.gz'):
            with self.subTest(image=image):
                self.row['image'] = image
                self.save_manifest()
                with self.assertRaises(ValueError):
                    self.run_archive(restore=True)
        self.assertFalse(self.target.exists())

    def test_restore_is_selected_idempotent_and_honest_without_historical_hashes(self):
        self.make_archive(extra=[('amos22/imagesTs/amos_0500.nii.gz', b'no')])
        audit = self.run_archive(restore=True)
        for item in audit['files']:
            path = self.target / item['target_relative']
            self.assertEqual(item['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(item['zip_crc_verified'])
            self.assertFalse(item['historical_sha256_verified'])
            self.assertFalse(item['historical_mtime_restored'])
        self.assertEqual((self.target/'imagesVa/imagesVa/amos_0008.nii.gz').read_bytes(), self.image)
        self.assertFalse((self.target/'imagesTs').exists())
        again = self.run_archive(restore=True)
        self.assertEqual({f['status'] for f in again['files']}, {'reused'})

    def test_existing_different_target_is_preserved(self):
        self.make_archive()
        path = self.target/'imagesVa/imagesVa/amos_0008.nii.gz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'newer different local CT')
        with self.assertRaisesRegex(ValueError, 'existing'):
            self.run_archive(restore=True)
        self.assertEqual(path.read_bytes(), b'newer different local CT')

    def historical(self, sha=None, size=None, mtime=1_700_000_000_123_456_700):
        path = self.base/'protocol.json'
        key = '../../amos22/imagesVa/imagesVa/amos_0008.nii.gz'
        path.write_text(json.dumps(dict(data=dict(
            dataset={key: sha or hashlib.sha256(self.image).hexdigest()},
            stats={key: dict(size=len(self.image) if size is None else size, mtime_ns=mtime)}))))
        return path, mtime

    def test_historical_sha_and_size_bind_mtime_without_inventing_label_history(self):
        self.make_archive()
        protocol, recorded_mtime = self.historical()
        audit = self.run_archive(restore=True, historical_protocol=protocol)
        image, label = audit['files']
        self.assertTrue(image['historical_sha256_verified'])
        self.assertTrue(image['historical_mtime_restored'])
        self.assertEqual((self.target/image['target_relative']).stat().st_mtime_ns, recorded_mtime)
        self.assertFalse(label['historical_sha256_verified'])
        self.assertFalse(label['historical_mtime_restored'])
        self.assertIn('official_md5', label['integrity_provenance'])
        self.assertNotIn('historical_sha256', label['integrity_provenance'])

    def test_wrong_historical_sha_or_size_never_publishes_the_file(self):
        self.make_archive()
        for kwargs in (dict(sha='0'*64), dict(size=999)):
            with self.subTest(kwargs=kwargs):
                protocol, _ = self.historical(**kwargs)
                with self.assertRaisesRegex(ValueError, 'historical'):
                    self.run_archive(restore=True, historical_protocol=protocol)
                self.assertFalse((self.target/'imagesVa/imagesVa/amos_0008.nii.gz').exists())

    def test_selected_size_reserve_is_checked_before_creating_target_root(self):
        self.make_archive()
        available = recovery.RESERVE_BYTES + len(self.image) + len(self.label) - 1
        with patch.object(recovery.shutil, 'disk_usage', return_value=SimpleNamespace(free=available)):
            inspection = self.run_archive()
            self.assertFalse(inspection['space_sufficient'])
            with self.assertRaisesRegex(ValueError, 'space|reserve'):
                self.run_archive(restore=True)
        self.assertFalse(self.target.exists())

    def test_per_file_reserve_is_checked_again_if_free_space_falls(self):
        self.make_archive()
        ample = SimpleNamespace(free=recovery.RESERVE_BYTES + 100_000)
        low = SimpleNamespace(free=recovery.RESERVE_BYTES)
        with patch.object(recovery.shutil, 'disk_usage', side_effect=[ample, low]):
            with self.assertRaisesRegex(ValueError, 'space|reserve'):
                self.run_archive(restore=True)
        self.assertFalse((self.target/'imagesVa/imagesVa/amos_0008.nii.gz').exists())

    def test_crc_corruption_is_caught_independently_of_whole_archive_hash(self):
        self.make_archive()
        with zipfile.ZipFile(self.archive) as z:
            member = z.infolist()[0]
            start = member.header_offset + 30 + len(member.filename.encode()) + len(member.extra)
        corrupt = bytearray(self.archive.read_bytes())
        corrupt[start] ^= 1
        self.archive.write_bytes(corrupt)
        with self.assertRaisesRegex(zipfile.BadZipFile, 'CRC'):
            self.run_archive(restore=True)
        self.assertFalse((self.target/'imagesVa/imagesVa/amos_0008.nii.gz').exists())

    def test_interruption_leaves_own_partial_and_retry_preserves_all_other_files(self):
        self.make_archive()
        with patch.object(recovery.os, 'fsync', side_effect=OSError('simulated interruption')):
            with self.assertRaisesRegex(OSError, 'simulated interruption'):
                self.run_archive(restore=True)
        parent = self.target/'imagesVa/imagesVa'
        partials = list(parent.glob('*.part'))
        self.assertEqual(len(partials), 1)
        evidence = partials[0].read_bytes()
        sentinel = parent/'.amos_0008.nii.gz.restore-somebody-else.part'
        sentinel.write_bytes(b'keep this unrelated file')
        self.run_archive(restore=True)
        self.assertEqual(partials[0].read_bytes(), evidence)
        self.assertEqual(sentinel.read_bytes(), b'keep this unrelated file')
        self.assertEqual((parent/'amos_0008.nii.gz').read_bytes(), self.image)

    def test_atomic_publication_refuses_concurrent_target_creation(self):
        self.make_archive()
        operation = 'rename' if os.name == 'nt' else 'link'
        original = getattr(os, operation)

        def another_writer(source, target):
            Path(target).write_bytes(b'concurrent writer')
            return original(source, target)

        with patch.object(recovery.os, operation, side_effect=another_writer):
            with self.assertRaises(FileExistsError):
                self.run_archive(restore=True)
        self.assertEqual((self.target/'imagesVa/imagesVa/amos_0008.nii.gz').read_bytes(),
                         b'concurrent writer')

    def test_target_symlink_cannot_redirect_writes_outside_target_root(self):
        self.make_archive()
        outside = self.base/'outside'
        outside.mkdir()
        self.target.mkdir()
        try:
            (self.target/'imagesVa').symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest('symlink creation unavailable: ' + str(exc))
        with self.assertRaisesRegex(ValueError, 'symlink|escape'):
            self.run_archive(restore=True)
        self.assertFalse(list(outside.iterdir()))

    def test_relative_arguments_resolve_from_project_root(self):
        self.make_archive()
        with patch.object(recovery, 'ROOT', self.base), \
             patch.object(recovery, 'OFFICIAL_SIZE', self.archive.stat().st_size), \
             patch.object(recovery, 'OFFICIAL_MD5', hashlib.md5(self.archive.read_bytes()).hexdigest()):
            report = recovery.process_archive('amos22.zip', 'split.json', target_root='amos22')
        self.assertEqual(Path(report['target_root']), self.target)

    def test_cli_emits_exclusive_audit_and_refuses_reusing_audit_before_writes(self):
        self.assertTrue(callable(getattr(recovery, 'main', None)), 'provide inspect/restore CLI')
        self.make_archive()
        audit = self.base/'audit.json'
        args = ['inspect', '--archive', str(self.archive), '--manifest', str(self.manifest),
                '--target-root', str(self.target), '--audit', str(audit)]
        with patch.object(recovery, 'OFFICIAL_SIZE', self.archive.stat().st_size), \
             patch.object(recovery, 'OFFICIAL_MD5', hashlib.md5(self.archive.read_bytes()).hexdigest()), \
             patch('sys.stdout', new_callable=io.StringIO), patch('sys.stderr', new_callable=io.StringIO):
            self.assertEqual(recovery.main(args), 0)
            saved = audit.read_bytes()
            self.assertEqual(json.loads(saved)['mode'], 'inspect')
            args[0] = 'restore'
            self.assertEqual(recovery.main(args), 2)
        self.assertEqual(audit.read_bytes(), saved)
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
