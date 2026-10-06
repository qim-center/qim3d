import qim3d


def test_starting_class():
    app = qim3d.gui.annotation_tool.Interface()

    assert app.title == "Annotation Tool"
