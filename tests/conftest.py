"""Test configuration and fixtures"""

import pytest


@pytest.fixture
def dummy_fixture():
    """Dummy fixture for testing"""
    return "test_value"
