import qim3d


def some_function(a: int, b: int) -> int:
    """Docstring."""
    a = 1
    b = 2
    c = a + b

    print(f'my string {c}')

    # Detect blobs, and get binary mask
    blobs, mask = qim3d.processing.blob_detection(
        a,
        min_sigma=1,
        max_sigma=8,
        background='bright',
    )

    assert a == 1


class MyClass:
    def __init__(self):
        pass

    def wrong_function():
        """Docstring."""
