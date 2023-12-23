def inc(x):
    return x + 1


def test_answer():
    assert inc(inc(3)) == 5