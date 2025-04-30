import unittest
from nf_trace_db_manager import NextflowTraceDBManager


class TestNextflowTraceDBManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()

        # Create tables for testing
        self.db_manager.createTables()

    def tearDown(self):
        self.db_manager.close()

    def test_isConnected(self):
        self.assertTrue(self.db_manager.isConnected())

    def test_isDatabaseEmpty(self):
        self.assertFalse(self.db_manager.isDatabaseEmpty())  # Tables exist after creation

    def test_getUserVersion(self):
        version = self.db_manager.getUserVersion()
        self.assertEqual(version, 0)  # Default user_version is 0

    def test_createTables_force(self):
        self.db_manager.createTables(force=True)  # Recreate tables
        self.assertFalse(self.db_manager.isDatabaseEmpty())

    def test_close(self):
        self.db_manager.close()
        self.assertFalse(self.db_manager.isConnected())


if __name__ == "__main__":
    unittest.main()