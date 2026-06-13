import unittest
from types import SimpleNamespace

from lafla_ai_core.training.distributed import (
    initialize_distributed,
    read_distributed_environment,
    resolve_distributed_backend,
)


class _FakeCuda:
    def __init__(self, device_count: int = 2) -> None:
        self._device_count = device_count
        self.selected_devices: list[int] = []

    def device_count(self) -> int:
        return self._device_count

    def set_device(self, local_rank: int) -> None:
        self.selected_devices.append(local_rank)


class _FakeDistributed:
    class ReduceOp:
        SUM = "sum"

    def __init__(self, *, nccl_available: bool = True) -> None:
        self.nccl_available = nccl_available
        self.init_calls: list[dict[str, object]] = []
        self.barrier_calls = 0
        self.destroy_calls = 0

    def is_available(self) -> bool:
        return True

    def is_nccl_available(self) -> bool:
        return self.nccl_available

    def init_process_group(self, **kwargs) -> None:
        self.init_calls.append(kwargs)

    def barrier(self) -> None:
        self.barrier_calls += 1

    def destroy_process_group(self) -> None:
        self.destroy_calls += 1


class DistributedEnvironmentTest(unittest.TestCase):
    def test_reads_complete_torchrun_environment(self):
        environment = read_distributed_environment(
            {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}
        )

        self.assertTrue(environment.is_distributed)
        self.assertFalse(environment.is_primary)
        self.assertEqual(environment.world_size, 2)
        self.assertEqual(environment.rank, 1)
        self.assertEqual(environment.local_rank, 1)

    def test_empty_environment_is_single_process(self):
        environment = read_distributed_environment({})

        self.assertFalse(environment.is_distributed)
        self.assertTrue(environment.is_primary)
        self.assertEqual(environment.world_size, 1)
        self.assertEqual(environment.rank, 0)
        self.assertEqual(environment.local_rank, 0)

    def test_incomplete_torchrun_environment_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE.*RANK.*LOCAL_RANK"):
            read_distributed_environment({"WORLD_SIZE": "2", "RANK": "0"})

    def test_auto_backend_resolves_cuda_to_nccl(self):
        self.assertEqual(resolve_distributed_backend("auto", "cuda"), "nccl")
        self.assertEqual(resolve_distributed_backend("auto", "cpu"), "gloo")


class DistributedRuntimeTest(unittest.TestCase):
    def test_single_process_does_not_initialize_process_group(self):
        distributed = _FakeDistributed()
        torch_module = SimpleNamespace(cuda=_FakeCuda(), distributed=distributed)

        runtime = initialize_distributed(
            configured_backend="auto",
            device_type="cuda",
            environ={},
            torch_module=torch_module,
        )

        self.assertFalse(runtime.is_distributed)
        self.assertTrue(runtime.is_primary)
        self.assertEqual(distributed.init_calls, [])

    def test_multi_process_initializes_nccl_on_local_rank(self):
        cuda = _FakeCuda()
        distributed = _FakeDistributed()
        torch_module = SimpleNamespace(cuda=cuda, distributed=distributed)

        runtime = initialize_distributed(
            configured_backend="auto",
            device_type="cuda",
            environ={"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
            torch_module=torch_module,
        )

        self.assertTrue(runtime.is_distributed)
        self.assertFalse(runtime.is_primary)
        self.assertEqual(cuda.selected_devices, [1])
        self.assertEqual(
            distributed.init_calls,
            [{"backend": "nccl", "init_method": "env://"}],
        )

        runtime.barrier()
        runtime.close()
        runtime.close()

        self.assertEqual(distributed.barrier_calls, 1)
        self.assertEqual(distributed.destroy_calls, 1)

    def test_nccl_unavailable_fails_before_initialization(self):
        torch_module = SimpleNamespace(
            cuda=_FakeCuda(),
            distributed=_FakeDistributed(nccl_available=False),
        )

        with self.assertRaisesRegex(RuntimeError, "NCCL"):
            initialize_distributed(
                configured_backend="auto",
                device_type="cuda",
                environ={"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"},
                torch_module=torch_module,
            )


if __name__ == "__main__":
    unittest.main()
