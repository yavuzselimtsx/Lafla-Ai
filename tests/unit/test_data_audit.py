import unittest

from lafla_ai_core.data.audit import audit_manifest
from lafla_ai_core.data.manifest import DatasetManifest


class DataAuditTest(unittest.TestCase):
    def test_accepts_reviewed_manifest(self):
        manifest = DatasetManifest.from_mapping(
            {
                "datasetVersion": "lafla-test",
                "targetTokens": 1000,
                "policy": {
                    "unknownLicenseAllowed": False,
                    "piiRequiredForInstructionAndPreference": True,
                    "syntheticDataRequiresTeacherAndSource": True,
                    "minimumTurkishConversationTokens": 0,
                    "minimumEvaluationSets": 1,
                },
                "sources": [
                    {
                        "sourceId": "owned_tr",
                        "loader": "local_jsonl",
                        "subset": "default",
                        "language": "tr",
                        "license": "lafla-owned",
                        "weight": 1.0,
                        "usage": "instruction",
                        "trustTier": "owned",
                        "sourceUrl": "file://owned",
                        "piiCleaned": True,
                    }
                ],
                "evaluationSets": [
                    {
                        "sourceId": "eval_tr",
                        "loader": "local_jsonl",
                        "language": "tr",
                        "usage": "dialogue_eval",
                        "sourceUrl": "file://eval",
                    }
                ],
            }
        )
        report = audit_manifest(manifest)
        self.assertTrue(report.ok)

    def test_rejects_unknown_license_and_missing_pii(self):
        manifest = DatasetManifest.from_mapping(
            {
                "datasetVersion": "lafla-test",
                "targetTokens": 1000,
                "policy": {
                    "unknownLicenseAllowed": False,
                    "piiRequiredForInstructionAndPreference": True,
                    "syntheticDataRequiresTeacherAndSource": True,
                    "minimumTurkishConversationTokens": 0,
                    "minimumEvaluationSets": 1,
                },
                "sources": [
                    {
                        "sourceId": "bad",
                        "loader": "local_jsonl",
                        "language": "tr",
                        "license": "unknown",
                        "weight": 1.0,
                        "usage": "instruction",
                        "trustTier": "owned",
                        "sourceUrl": "",
                    }
                ],
                "evaluationSets": [],
            }
        )
        report = audit_manifest(manifest)
        self.assertFalse(report.ok)
        codes = {finding.code for finding in report.findings}
        self.assertIn("unknown_license", codes)
        self.assertIn("pii_cleaning_required", codes)
        self.assertIn("source_url_missing", codes)


if __name__ == "__main__":
    unittest.main()

