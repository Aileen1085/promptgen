from pathlib import Path
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'run_train_v10_2_ct13_e490_lowpg_lr_control_10.sh'


class LowPromptLRLauncherTests(unittest.TestCase):
    def test_launcher_exists(self):
        self.assertTrue(LAUNCHER.is_file(), 'isolated low-PromptGen LR launcher missing')

    def test_only_authorized_overrides(self):
        if not LAUNCHER.is_file():
            self.fail('isolated low-PromptGen LR launcher missing')
        source = LAUNCHER.read_text(encoding='utf-8')
        command = source[source.index('exec bash '):].replace('\\\n', ' ')
        self.assertEqual(shlex.split(command), [
            'exec', 'bash', '$ROOT/run_train_v10_2_ct13_e490_full_decoder_40.sh',
            '--epochs', '10', '--lr-schedule-epochs', '40', '--validate-every', '5',
            '--prompt-lr', '2.5e-7', '--adapter-lr', '5e-7',
            '--memory-adapter-lr', '5e-7', '$@'])
        self.assertIn('output/v10_2_ct13_e490_lowpg_lr_control_10_20261004', source)
        self.assertIn('export GPU="${GPU:-1}"', source)
        self.assertNotIn('--resume', command)
        self.assertNotIn('/home/', source)


if __name__ == '__main__':
    unittest.main()
