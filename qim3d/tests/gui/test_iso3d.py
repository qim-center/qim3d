import qim3d
import multiprocessing
import time


def test_starting_class():
    app = qim3d.gui.iso3d.Interface()

    assert app.title == "Isosurfaces for 3D visualization"


def test_app_launch():
    ip = "0.0.0.0"
    port = 65432

    def start_server(ip, port):
        app = qim3d.gui.iso3d.Interface()
        app.launch(server_name=ip, server_port=port)

    proc = multiprocessing.Process(target=start_server, args=(ip, port))
    proc.start()

    # App is running in a separate process
    # So we try to get a response for a while
    max_checks = 5
    check = 0
    server_running = False
    while check < max_checks and not server_running:
        server_running = qim3d.utils.internal_tools.is_server_running(ip, port)
        time.sleep(1)
        check += 1

    # Terminate tre process before assertions
    proc.terminate()

    assert server_running is True
