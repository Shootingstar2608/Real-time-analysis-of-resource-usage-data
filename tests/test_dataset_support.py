import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import csv
import gzip
import os
import tempfile
import unittest

from data.dataset_utils import (
    detect_dataset_type,
    records_to_dataframe,
    load_aggregated_series_by_timestamp,
)


class DatasetSupportTests(unittest.TestCase):
    def test_detect_and_normalize_alibaba_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usage_path = os.path.join(tmpdir, "machine_usage_sample.csv")
            meta_path = os.path.join(tmpdir, "machine_meta_sample.csv")

            with open(meta_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["m1", 0, 1, "fd-a", 64, 96, "working"])
                writer.writerow(["m2", 0, 2, "fd-b", 32, 48, "idle"])

            with open(usage_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["m1", 0, 10, 20, 1.5, 12, 5.0, 2.0, 7.0])
                writer.writerow(["m2", 0, 40, 35, 2.5, 9, 1.0, 3.0, 6.0])
                writer.writerow(["m1", 60, 15, 25, 1.0, 15, 4.0, 1.0, 5.0])

            self.assertEqual(detect_dataset_type(tmpdir), "alibaba")
            df = records_to_dataframe(tmpdir, dataset="alibaba")
            self.assertEqual(len(df), 3)
            self.assertIn("avg_memory", df.columns)
            self.assertIn("network_in_mbps", df.columns)
            self.assertEqual(df.iloc[0]["vm_category"], "working")
            self.assertEqual(df.iloc[0]["vm_core_count"], 64)
            self.assertEqual(df.iloc[0]["vm_memory_gb"], 96)
            self.assertEqual(df.iloc[0]["avg_cpu"], 10.0)

            agg_df = load_aggregated_series_by_timestamp(tmpdir, dataset="alibaba")
            self.assertGreaterEqual(len(agg_df), 2)
            self.assertIn("avg_memory", agg_df.columns)

    def test_detect_and_normalize_azure_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vmtable_path = os.path.join(tmpdir, "vmtable.csv.gz")
            cpu_path = os.path.join(tmpdir, "vm_cpu_readings-file-1-of-195.csv.gz")

            with gzip.open(vmtable_path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["vm1", "sub", "dep", 0, 100, 90, 50, 80, "Interactive", 8, 16])

            with gzip.open(cpu_path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([0, "vm1", 10, 20, 15])
                writer.writerow([300, "vm1", 20, 40, 30])

            self.assertEqual(detect_dataset_type(tmpdir), "azure")
            df = records_to_dataframe(tmpdir, dataset="azure")
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["vm_category"], "Interactive")
            self.assertEqual(df.iloc[1]["cpu_range"], 20.0)
            self.assertEqual(df.iloc[0]["avg_memory"], 0.0)


if __name__ == "__main__":
    unittest.main()
