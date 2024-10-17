from fastapi import FastAPI
import qim3d.utils
from . import data_explorer
from . import iso3d
from . import local_thickness
from . import annotation_tool
from . import layers2d
from .qim_theme import QimTheme


def run_gradio_app(gradio_interface, host="0.0.0.0"):
    import gradio as gr
    import uvicorn

    # Get port using the QIM API
    port_dict = qim3d.utils.get_port_dict()

    if "gradio_port" in port_dict:
        port = port_dict["gradio_port"]
    elif "port" in port_dict:
        port = port_dict["port"]
    else:
        raise Exception("Port not specified from QIM API")

    qim3d.utils.gradio_header(gradio_interface.title, port)

    # Create FastAPI with mounted gradio interface
    app = FastAPI()
    path = f"/gui/{port_dict['username']}/{port}/"
    app = gr.mount_gradio_app(app, gradio_interface, path=path)

    # Full path
    print(f"http://{host}:{port}{path}")

    # Run the FastAPI server usign uvicorn
    uvicorn.run(app, host=host, port=int(port))
