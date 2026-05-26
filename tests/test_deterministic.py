import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import functions under test
from pipeline.build_dataset import _coerce_bool, clean_state_bill_title, _normalize_historical_bills
from app import calculate_cheap_pca
from pipeline.policy_matching import AnalogMatcher


class TestDeterministicLogic(unittest.TestCase):
    """Unit tests for deterministic mathematical and processing functions."""

    def test_coerce_bool(self):
        """Verify type coercion to robust boolean values."""
        self.assertTrue(_coerce_bool(True))
        self.assertTrue(_coerce_bool("true"))
        self.assertTrue(_coerce_bool("yes"))
        self.assertTrue(_coerce_bool("1"))
        self.assertFalse(_coerce_bool(False))
        self.assertFalse(_coerce_bool("false"))
        self.assertFalse(_coerce_bool("0"))
        self.assertFalse(_coerce_bool(None))

    def test_clean_state_bill_title(self):
        """Verify legal preambles and citations are correctly stripped."""
        raw_title_1 = "An Act providing for tax credits for local manufacturers (P.L. 12, No. 34)"
        cleaned_1 = clean_state_bill_title(raw_title_1)
        self.assertNotIn("An Act", cleaned_1)
        self.assertNotIn("P.L.", cleaned_1)
        self.assertIn("tax credits", cleaned_1)

        raw_title_2 = "A Bill relating to post-secondary education voucher funding."
        cleaned_2 = clean_state_bill_title(raw_title_2)
        self.assertNotIn("relating to", cleaned_2)
        self.assertIn("post-secondary", cleaned_2)

    def test_vectorized_normalize_historical_bills(self):
        """Verify raw policy csv records are vectorized and normalized correctly."""
        raw_df = pd.DataFrame([
            {
                "bill_id": "HB-101",
                "title": "An Act relating to solar subsidies",
                "state": "California",
                "enacted_date": "2023-05-15",
                "text": "Solar energy tax credits for local households."
            },
            {
                "bill_id": "",
                "title": "Ceremonial post office naming",
                "state": "United States",
                "introduced_date": "2021-08-10",
                "text": "Commemorative plaque for federal offices."
            }
        ])
        
        normalized = _normalize_historical_bills(raw_df)
        self.assertEqual(len(normalized), 2)
        
        # Verify columns exist
        self.assertIn("jurisdiction", normalized.columns)
        self.assertIn("level", normalized.columns)
        self.assertIn("state_code", normalized.columns)
        self.assertIn("session_year", normalized.columns)
        
        # Verify California row is state-level and has clean title
        cali_row = normalized[normalized["state_code"] == "CA"].iloc[0]
        self.assertEqual(cali_row["jurisdiction"], "california")
        self.assertEqual(cali_row["level"], "state")
        self.assertEqual(cali_row["session_year"], 2023)
        self.assertNotIn("An Act relating to", cali_row["title"])

        # Verify Federal row is federal-level and got assigned a dynamic ID
        fed_row = normalized[normalized["state_code"] == "US"].iloc[0]
        self.assertEqual(fed_row["jurisdiction"], "federal")
        self.assertEqual(fed_row["level"], "federal")
        self.assertEqual(fed_row["session_year"], 2021)
        self.assertTrue(fed_row["bill_id"].startswith("hist_"))

    def test_calculate_cheap_pca_sign_stability(self):
        """Verify Singular Value Decomposition sign-locking (svd_flip) is mathematically stable."""
        # Generate simple deterministic vector dataset
        np.random.seed(42)
        vectors = np.random.randn(10, 5)
        
        # Perform SVD PCA twice, once with vectors and once with slightly scaled vectors
        coords_1 = calculate_cheap_pca(vectors, num_components=2)
        coords_2 = calculate_cheap_pca(vectors * 1.5, num_components=2)
        
        # The scale will change, but the sign of coordinates must be identical (no coordinate flipping/mirroring)
        sign_1_col1 = np.sign(coords_1[:, 0])
        sign_2_col1 = np.sign(coords_2[:, 0])
        np.testing.assert_array_equal(sign_1_col1, sign_2_col1)

    @patch("pipeline.policy_matching.requests.post")
    def test_analog_matcher_precomputed_search(self, mock_post):
        """Verify AnalogMatcher loads cached embeddings and performs precomputed BLAS searches."""
        # Setup mock embeddings response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_post.return_value = mock_response

        # Mock historical dataset
        hist_df = pd.DataFrame([
            {
                "bill_id": "hist_1",
                "title": "Corporate tax credit",
                "state": "California",
                "jurisdiction": "california",
                "level": "state"
            },
            {
                "bill_id": "hist_2",
                "title": "Maternal health clinics",
                "state": "United States",
                "jurisdiction": "federal",
                "level": "federal"
            }
        ])

        # Pre-seed embedding cache so initialization doesn't call API
        with patch.object(AnalogMatcher, "_build_embedding_cache") as mock_cache:
            matcher = AnalogMatcher(historical_bills_df=hist_df)
            matcher.embedding_cache = {
                "hist_1": np.array([0.5] * 768, dtype=float),
                "hist_2": np.array([-0.5] * 768, dtype=float)
            }
            # Manually trigger precomputation
            matcher._precompute_matrices()

        self.assertEqual(len(matcher.bill_ids), 2)
        self.assertIsNotNone(matcher.embeddings_matrix)
        self.assertEqual(matcher.embeddings_matrix.shape, (2, 768))

        # Perform a search for a state proposal
        target = {
            "title": "Solar incentives",
            "jurisdiction": "california",
            "level": "state"
        }
        
        analogs = matcher.find_similar_bills(target, min_threshold=0.5)
        
        # Verify matching coordinates and dynamic fallbacks evaluated successfully
        self.assertTrue(len(analogs) > 0)
        self.assertEqual(analogs[0]["bill_id"], "hist_1")


if __name__ == "__main__":
    unittest.main()
