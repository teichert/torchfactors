from wrap.resolvers import last, safe


def test_safe():
    assert safe("test") == "test"
    assert safe("test_this") == "test_this"
    assert safe("test this") == "test-this"
    assert safe("test this too") == "test-this-too"
    assert safe("test/this too") == "test-this-too"
    assert safe("test/this too") == "test-this-too"


def test_last():
    assert last("test", 's') == "t"
    assert last("test_this", '_') == "this"
    assert last("test this", ' ') == "this"
    assert last("test this too", ' ') == "too"
    assert last("test/this too", ' ') == "too"
    assert last("test/this too", '/') == "this too"
