import importlib
import unittest
from types import SimpleNamespace


class ProbeBindingTests(unittest.TestCase):
    def test_compiled_main_uses_current_decode_binding(self):
        old=lambda:'old'
        current=lambda:'native-twin-checked'
        space={'_dynamic_adaptive_decode':old}
        exec('def main(): return _dynamic_adaptive_decode()',space)
        evaluator=SimpleNamespace(main=space['main'],_dynamic_adaptive_decode=current)
        self.assertEqual(evaluator.main(),'old')
        module=importlib.import_module('tools.bone_shared_floor_probe_runtime')
        module.bind_current_decode(evaluator)
        self.assertEqual(evaluator.main(),'native-twin-checked')

    def test_new_engine_process_and_source_guards(self):
        m=importlib.import_module('tools.bone_shared_floor_probe_grid').engine()
        self.assertTrue(m.OUT.endswith('_restorefix_probebinding'))
        self.assertTrue(m.__file__.endswith('bone_shared_floor_probe_grid.py'))
        self.assertIn('tools/bone_shared_floor_probe_runtime.py',m.SOURCES)
        self.assertIn('tools.bone_shared_floor_probe_runtime',m.run.__code__.co_names)


if __name__=='__main__':unittest.main()
