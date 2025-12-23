def test_import():
    """Test that the package can be imported."""
    import sharp
    assert sharp is not None


def test_cli_import():
    """Test that CLI modules can be imported."""
    from sharp import cli
    assert cli is not None

