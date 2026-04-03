import unittest
from pathlib import Path


class OptionalImportsAreLazyTest(unittest.TestCase):
    def test_data_loader_does_not_require_hf_or_sktime_at_module_import_time(self):
        content = Path("data_provider/data_loader.py").read_text()
        self.assertNotIn("from sktime.datasets import load_from_tsfile_to_dataframe", content)
        self.assertNotIn("from datasets import load_dataset", content)
        self.assertNotIn("from huggingface_hub import hf_hub_download", content)
        self.assertIn("def _load_dataset_from_hf(", content)
        self.assertIn("def _load_tsfile_dataframe(", content)
        self.assertIn("def _hf_hub_download(", content)

    def test_self_attention_family_does_not_require_reformer_at_module_import_time(self):
        content = Path("layers/SelfAttention_Family.py").read_text()
        self.assertNotIn("from reformer_pytorch import LSHSelfAttention", content)
        self.assertIn("def _get_lsh_self_attention(", content)


if __name__ == "__main__":
    unittest.main()
