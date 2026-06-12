import unittest

from lafla_ai_core.config.loader import load_mapping


class DataSourcePlanTest(unittest.TestCase):
    def test_100m_source_plan_matches_language_and_capability_mix(self):
        plan = load_mapping("configs/data/lafla-100m-source-plan.json")["source_plan"]
        sources = plan["sources"]
        self.assertEqual(plan["dataset_version"], "lafla-100m-thinking-realdata-2026-06")
        self.assertAlmostEqual(sum(float(source["weight"]) for source in sources), 1.0)
        by_domain: dict[str, float] = {}
        for source in sources:
            by_domain[source["domain"]] = by_domain.get(source["domain"], 0.0) + float(source["weight"])
        expected = {
            "turkish_general": 0.42,
            "german_general": 0.23,
            "english_general": 0.10,
            "tr_de_encyclopedic_history": 0.10,
            "math_logic_science": 0.08,
            "code_technical": 0.07,
        }
        self.assertEqual(set(by_domain), set(expected))
        for domain, weight in expected.items():
            self.assertAlmostEqual(by_domain[domain], weight)
        risky = [source for source in sources if source["review_state"] == "review_required"]
        self.assertTrue(risky)
        self.assertTrue(all(source["primary_eligible"] is False for source in risky))

    def test_source_plan_weights_sum_to_one_and_ids_are_unique(self):
        plan = load_mapping("configs/data/lafla-400m-source-plan.json")["source_plan"]
        self.assertEqual(plan["dataset_version"], "lafla-400m-thinking-2026-06")
        self.assertEqual(plan["target_steps"], 4000)
        sources = plan["sources"]
        self.assertAlmostEqual(sum(float(source["weight"]) for source in sources), 1.0)
        ids = [source["source_id"] for source in sources]
        self.assertEqual(len(ids), len(set(ids)))

    def test_380m_source_plan_covers_core_capability_domains(self):
        plan = load_mapping("configs/data/lafla-380m-source-plan.json")["source_plan"]
        self.assertEqual(plan["dataset_version"], "lafla-380m-thinking-realdata-2026-06")
        self.assertEqual(plan["target_steps"], 50_000)
        sources = plan["sources"]
        self.assertAlmostEqual(sum(float(source["weight"]) for source in sources), 1.0)
        domains = {source["domain"] for source in sources}
        self.assertGreaterEqual(
            domains,
            {"turkish", "english", "math", "code", "cybersecurity", "identity"},
        )
        self.assertTrue(all(source["data_kind"] == "real_or_owned" for source in sources))

    def test_review_required_sources_are_not_marked_primary(self):
        plan = load_mapping("configs/data/lafla-400m-source-plan.json")["source_plan"]
        risky = [source for source in plan["sources"] if "review" in source["status"]]
        self.assertTrue(risky)
        self.assertTrue(all(source["status"] != "primary" for source in risky))


if __name__ == "__main__":
    unittest.main()
