"""Content binding and isolation checks for the in-memory raw-logit memo."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.bone_shared_logits import TensorMemo, cpu_clone, tensor_bytes, tensor_digest, to_device


class DigestTests(unittest.TestCase):
    def test_mapping_order_does_not_change_digest(self):
        a = {"image": torch.arange(6).reshape(2, 3), "prompt": [1, {"z": 2, "a": 3}]}
        b = {"prompt": [1, {"a": 3, "z": 2}], "image": a["image"].clone()}
        self.assertEqual(tensor_digest(a), tensor_digest(b))

    def test_primitive_and_container_types_are_distinct(self):
        values = [None, False, 0, 0.0, "0", b"0", [0], (0,), {"0": 0}]
        self.assertEqual(len({tensor_digest(value) for value in values}), len(values))
        self.assertNotEqual(tensor_digest(0.0), tensor_digest(-0.0))

    def test_numpy_dtype_shape_and_contents_are_bound(self):
        a = np.arange(6, dtype=np.int32)
        variants = [a, a.astype(np.float32), a.reshape(2, 3), a + 1]
        self.assertEqual(len({tensor_digest(value) for value in variants}), len(variants))

    def test_torch_numpy_and_scalar_types_are_distinct(self):
        self.assertNotEqual(tensor_digest(torch.tensor([1], dtype=torch.int32)),
                            tensor_digest(np.array([1], dtype=np.int32)))
        self.assertNotEqual(tensor_digest(np.int64(1)), tensor_digest(1))
        self.assertNotEqual(tensor_digest(np.int32(1)), tensor_digest(np.int64(1)))

    def test_torch_dtype_shape_and_contents_are_bound(self):
        a = torch.arange(6, dtype=torch.int32)
        variants = [a, a.float(), a.reshape(2, 3), a + 1]
        self.assertEqual(len({tensor_digest(value) for value in variants}), len(variants))

    def test_logically_equal_noncontiguous_arrays_and_tensors_match(self):
        a = torch.arange(12).reshape(3, 4).T
        b = np.arange(12).reshape(3, 4).T
        self.assertEqual(tensor_digest(a), tensor_digest(a.contiguous()))
        self.assertEqual(tensor_digest(b), tensor_digest(np.ascontiguousarray(b)))
        self.assertEqual(tensor_digest(b[::-1]), tensor_digest(b[::-1].copy()))

    def test_bfloat16_scalar_empty_and_grad_tensors_hash_raw_values(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        self.assertEqual(tensor_digest(a), tensor_digest(a.clone()))
        self.assertNotEqual(tensor_digest(a), tensor_digest(a + 1))
        self.assertNotEqual(tensor_digest(a), tensor_digest(a.float()))
        for tensor in [torch.tensor(1.0), torch.empty(0), torch.ones(2, requires_grad=True)]:
            self.assertEqual(tensor_digest(tensor), tensor_digest(tensor.detach().clone()))

    def test_structured_numpy_dtypes_are_bound(self):
        a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])
        b = np.zeros(1, dtype=[("c", "i4"), ("d", "i4")])
        self.assertNotEqual(tensor_digest(a), tensor_digest(b))

    def test_unsupported_objects_and_object_arrays_are_rejected(self):
        for value in [object(), {1, 2}, np.array([object()], dtype=object)]:
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                tensor_digest(value)

    def test_recursive_containers_are_rejected(self):
        value = []
        value.append(value)
        with self.assertRaises(ValueError):
            tensor_digest(value)


class CloneTests(unittest.TestCase):
    def test_nested_clone_is_detached_cpu_and_has_independent_storage(self):
        tensor = torch.arange(6.0, requires_grad=True).reshape(2, 3).T
        array = np.arange(4, dtype=np.float32)[::-1]
        value = {"features": [tensor, (array, tensor)], "metadata": [None, "case", 1]}
        result = cpu_clone(value)
        self.assertEqual(tensor_digest(value), tensor_digest(result))
        self.assertIsInstance(result["features"], list)
        self.assertIsInstance(result["features"][1], tuple)
        self.assertEqual(result["features"][0].device.type, "cpu")
        self.assertFalse(result["features"][0].requires_grad)
        self.assertIsNone(result["features"][0].grad_fn)
        self.assertNotEqual(result["features"][0].data_ptr(), tensor.data_ptr())
        self.assertNotEqual(result["features"][0].data_ptr(), result["features"][1][1].data_ptr())
        self.assertFalse(np.shares_memory(result["features"][1][0], array))
        self.assertEqual(tensor_bytes(result), 2 * 6 * 4 + 4 * 4)

    def test_byte_count_includes_bfloat16_and_numpy_nested_leaves(self):
        value = [torch.zeros(5, dtype=torch.bfloat16), (np.zeros((2, 3), dtype=np.uint8), 7)]
        self.assertEqual(tensor_bytes(value), 16)
        self.assertEqual(tensor_bytes({"metadata": [None, "case", True, 1.0]}), 0)

    def test_to_device_preserves_tree_and_returns_fresh_storage(self):
        value = {"logits": torch.ones(2, dtype=torch.bfloat16), "aux": (np.ones(2), [3])}
        result = to_device(value, "cpu")
        self.assertEqual(tensor_digest(result), tensor_digest(value))
        result["logits"].zero_()
        result["aux"][0].fill(0)
        self.assertTrue(torch.equal(value["logits"], torch.ones(2, dtype=torch.bfloat16)))
        np.testing.assert_array_equal(value["aux"][0], np.ones(2))


class MemoTests(unittest.TestCase):
    def test_miss_preserves_original_and_hits_are_independent_detached_clones(self):
        memo = TensorMemo(max_bytes=1024)
        original = torch.tensor([1.0, 2.0], requires_grad=True) * 2
        first = memo.call("model-sha", {"image": torch.ones(2)}, lambda: original)
        second = memo.call("model-sha", {"image": torch.ones(2)}, lambda: self.fail("recomputed"))
        third = memo.call("model-sha", {"image": torch.ones(2)}, lambda: self.fail("recomputed"))
        self.assertIs(first, original)
        self.assertIsNot(second, original)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertNotEqual(second.data_ptr(), third.data_ptr())
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(second.requires_grad)
        self.assertIsNone(second.grad_fn)
        self.assertEqual(memo.stats, {"hits": 2, "misses": 1, "evictions": 0, "bytes": 8})

    def test_every_input_and_model_namespace_is_bound(self):
        base = {"pixels": torch.zeros(2), "prompt": np.array([1, 0]), "class_id": 10003,
                "settings": {"input_size": 1024, "window": 192, "use_fpn": True}}
        changes = [("pixels", torch.tensor([1.0, 0.0])), ("prompt", np.array([0, 1])),
                   ("class_id", 10004), ("settings", {**base["settings"], "window": 96})]
        for field, changed in changes:
            with self.subTest(field=field):
                memo = TensorMemo(1024)
                memo.call("model-a", base, lambda: torch.ones(1))
                modified = copy.deepcopy(base)
                modified[field] = changed
                result = memo.call("model-a", modified, lambda: torch.full((1,), 2.0))
                self.assertEqual(result.item(), 2)
                self.assertEqual(memo.stats["misses"], 2)
        memo = TensorMemo(1024)
        memo.call("model-a", base, lambda: torch.ones(1))
        result = memo.call("model-b", base, lambda: torch.full((1,), 3.0))
        self.assertEqual(result.item(), 3)
        self.assertEqual(memo.stats["misses"], 2)

    def test_original_and_hit_mutations_cannot_change_cache(self):
        memo = TensorMemo(1024)
        original = {"logits": torch.ones(3), "nested": [np.ones(2), {"value": [7]}]}
        expected = tensor_digest(original)
        memo.call("m", 1, lambda: original)
        original["logits"].zero_()
        original["nested"][0].fill(0)
        original["nested"][1]["value"].append(8)
        hit = memo.call("m", 1, lambda: self.fail("recomputed"))
        self.assertEqual(tensor_digest(hit), expected)
        hit["logits"].zero_()
        hit["nested"][0].fill(0)
        hit["nested"][1]["value"].append(9)
        self.assertEqual(tensor_digest(memo.call("m", 1, lambda: self.fail("recomputed"))), expected)

    def test_lru_evicts_least_recently_used_with_exact_bytes(self):
        memo = TensorMemo(8)
        for key in ["a", "b"]:
            memo.call("m", key, lambda: torch.ones(1, dtype=torch.int32))
        memo.call("m", "a", lambda: self.fail("recomputed a"))
        memo.call("m", "c", lambda: torch.ones(1, dtype=torch.int32))
        self.assertEqual(memo.stats, {"hits": 1, "misses": 3, "evictions": 1, "bytes": 8})
        memo.call("m", "b", lambda: torch.ones(1, dtype=torch.int32))
        self.assertEqual(memo.stats, {"hits": 1, "misses": 4, "evictions": 2, "bytes": 8})

    def test_oversize_bypass_preserves_existing_cache_and_original(self):
        memo = TensorMemo(4)
        memo.call("m", "small", lambda: torch.ones(1))
        large = torch.ones(2)
        for _ in range(2):
            self.assertIs(memo.call("m", "large", lambda: large), large)
        memo.call("m", "small", lambda: self.fail("small was evicted"))
        self.assertEqual(memo.stats, {"hits": 1, "misses": 3, "evictions": 0, "bytes": 4})
        self.assertFalse(memo.records[1]["cached"])

    def test_clear_frees_cache_and_preserves_counters_and_audit(self):
        memo = TensorMemo(4)
        memo.call("m", "a", lambda: torch.ones(1))
        memo.call("m", "a", lambda: self.fail("recomputed"))
        records = memo.records
        memo.clear()
        self.assertEqual(memo.stats, {"hits": 1, "misses": 1, "evictions": 0, "bytes": 0})
        self.assertEqual(memo.records, records)
        memo.call("m", "a", lambda: torch.ones(1))
        self.assertEqual(memo.stats["misses"], 2)

    def test_audit_records_bind_key_inputs_namespace_and_output_without_tensors(self):
        memo = TensorMemo(16)
        inputs = {"prompt": torch.ones(1), "pixels": np.zeros(2)}
        output = torch.ones(2, dtype=torch.bfloat16)
        memo.call({"model": "sha-a", "settings": [1]}, inputs, lambda: output)
        memo.call({"settings": [1], "model": "sha-a"}, dict(reversed(list(inputs.items()))),
                  lambda: self.fail("recomputed"))
        producer, hit = memo.records
        self.assertEqual(producer["event"], "producer")
        self.assertEqual(hit["event"], "hit")
        for field in ["key", "namespace_digest", "inputs_digest", "output_digest"]:
            self.assertEqual(len(producer[field]), 64)
            self.assertEqual(producer[field], hit[field])
        self.assertEqual(producer["output_digest"], tensor_digest(output))
        producer["output_digest"] = "changed"
        self.assertNotEqual(memo.records[0]["output_digest"], "changed")

    def test_cache_operations_do_not_write_disk(self):
        memo = TensorMemo(16)
        with patch("builtins.open", side_effect=AssertionError("disk access")):
            memo.call("m", [1], lambda: torch.ones(2))
            memo.call("m", [1], lambda: self.fail("recomputed"))
            memo.clear()

    def test_invalid_capacity_and_unsupported_outputs_fail_explicitly(self):
        for capacity in [-1, 1.5, True]:
            with self.subTest(capacity=capacity), self.assertRaises((TypeError, ValueError)):
                TensorMemo(capacity)
        with self.assertRaises(TypeError):
            TensorMemo(16).call("m", 1, lambda: object())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_producer_is_stored_only_as_detached_cpu_copy(self):
        memo = TensorMemo(32)
        original = torch.ones(2, device="cuda", requires_grad=True) * 2
        self.assertIs(memo.call("m", original, lambda: original), original)
        hit = memo.call("m", original.detach().cpu(), lambda: self.fail("recomputed"))
        self.assertEqual(hit.device.type, "cpu")
        self.assertFalse(hit.requires_grad)
        restored = to_device(hit, original.device)
        self.assertTrue(torch.equal(restored, original))
        self.assertNotEqual(restored.data_ptr(), original.data_ptr())


if __name__ == "__main__":
    unittest.main()
