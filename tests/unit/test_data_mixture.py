import unittest

from lafla_ai_core.data.manifest import SourceSpec
from lafla_ai_core.data.mixture import MixtureSource, build_mixture_plan, mixture_sources_from_manifest, normalize_weights


class DataMixtureTest(unittest.TestCase):
    def test_normalize_weights_preserves_ids_and_sums_to_one(self):
        sources = (
            MixtureSource("fineweb2_tr", "pretraining", 0.42),
            MixtureSource("wiki_tr", "pretraining", 0.06),
        )

        normalized = normalize_weights(sources)

        self.assertEqual([source.source_id for source in normalized], ["fineweb2_tr", "wiki_tr"])
        self.assertAlmostEqual(sum(source.normalized_weight for source in normalized), 1.0)
        self.assertGreater(normalized[0].normalized_weight, normalized[1].normalized_weight)

    def test_build_mixture_plan_filters_usage_and_overbuilds_sample_budget(self):
        sources = (
            MixtureSource("fineweb2_tr", "pretraining", 0.42),
            MixtureSource("wiki_tr", "pretraining", 0.06),
            MixtureSource("lafla_identity", "instruction", 0.08),
        )

        plan = build_mixture_plan(sources, usage="pretraining", total_samples=1000)

        self.assertEqual([entry.source_id for entry in plan.entries], ["fineweb2_tr", "wiki_tr"])
        self.assertEqual(sum(entry.sample_budget for entry in plan.entries), 1006)
        self.assertAlmostEqual(sum(entry.normalized_weight for entry in plan.entries), 1.0)

    def test_duplicate_source_id_fails_closed(self):
        sources = (
            MixtureSource("same", "pretraining", 0.5),
            MixtureSource("same", "instruction", 0.5),
        )

        with self.assertRaises(ValueError):
            normalize_weights(sources)

    def test_mixture_sources_from_manifest_specs_use_existing_manifest_contract(self):
        specs = (
            SourceSpec("fineweb2_tr", "hf", None, "tr", "ODC-By-1.0", 0.42, "pretraining", "trusted", "https://example.com"),
            SourceSpec("lafla_identity", "jsonl", None, "tr", "lafla-owned", 0.08, "instruction", "owned", "local"),
        )

        sources = mixture_sources_from_manifest(specs)

        self.assertEqual(sources[0], MixtureSource("fineweb2_tr", "pretraining", 0.42))
        self.assertEqual(sources[1].usage, "instruction")


if __name__ == "__main__":
    unittest.main()
