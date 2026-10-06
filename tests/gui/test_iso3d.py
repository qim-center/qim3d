import qim3d


def test_starting_class():
    app = qim3d.gui.iso3d.Interface()

    assert app.title == "Isosurfaces for 3D visualization"
