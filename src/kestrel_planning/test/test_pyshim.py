# pytest shim to run the unittest suite from test_waypoints.py
import unittest

# Import your unittest class from the existing file
from test_waypoints import TestPlanner

def test_unittest_suite():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPlanner)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful(), "Unittest suite failed — see stdout above for details"
