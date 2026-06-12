import unittest

from lafla_ai_core.runtime.rss import sample_process_tree


class _MemoryInfo:
    def __init__(self, rss: int, uss: int) -> None:
        self.rss = rss
        self.uss = uss


class _Process:
    def __init__(self, rss: int, uss: int, children=()) -> None:
        self._rss = rss
        self._uss = uss
        self._children = tuple(children)

    def children(self, recursive=True):
        return self._children

    def memory_info(self):
        return _MemoryInfo(self._rss, self._uss)

    def memory_full_info(self):
        return _MemoryInfo(self._rss, self._uss)


class _DisappearingProcess(_Process):
    def memory_info(self):
        raise LookupError("process exited")


class ProcessTreeRssTest(unittest.TestCase):
    def test_sample_sums_parent_and_all_child_process_memory(self):
        process = _Process(
            100,
            80,
            children=(
                _Process(40, 30),
                _Process(20, 10),
            ),
        )

        sample = sample_process_tree(process)

        self.assertEqual(sample.rss_bytes, 160)
        self.assertEqual(sample.uss_bytes, 120)
        self.assertEqual(sample.process_count, 3)

    def test_sample_ignores_child_that_exits_during_measurement(self):
        process = _Process(100, 80, children=(_DisappearingProcess(40, 30),))
        sample = sample_process_tree(process)
        self.assertEqual(sample.rss_bytes, 100)
        self.assertEqual(sample.process_count, 1)


if __name__ == "__main__":
    unittest.main()
