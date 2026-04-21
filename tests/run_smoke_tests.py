"""Convenience entrypoint for Colab/local smoke tests."""

import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
