import qim3d


def test_memory():
    mem = qim3d.utils.Memory()

    assert all([mem.total > 0, mem.used > 0, mem.free > 0])

    assert mem.used + mem.free == mem.total
