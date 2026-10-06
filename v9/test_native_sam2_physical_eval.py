import ast
import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / 'tools/native_sam2_physical_eval.py'


def test_native_runner_exists_and_has_no_promptgen_or_adapter_import():
    assert PATH.is_file(), 'native SAM2 runner is missing'
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or '')
    assert not any('promptgen' in name.lower() or 'prompt_token_generator' in name or 'v10_2' in name for name in modules)
    assert 'sam2.build_sam' in modules


def test_cli_defaults_original_weights_and_physical_evidence():
    assert PATH.is_file(), 'native SAM2 runner is missing'
    spec = importlib.util.spec_from_file_location('native_eval_cli', PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = module.parse_args(['--job', 'job.json'])
    assert args.sigma_mm == 2. and args.amplitude == 4. and args.max_points == 16
    assert args.checkpoint == 'sam2/checkpoints/sam2.1_hiera_large.pt'
    assert args.dense_prior == 'gaussian'
