from src.lib.some_library import add


def test_add():
    # Test case 1: Check if add(2, 3) returns 5
    result = add(2, 3)
    assert result == 5
