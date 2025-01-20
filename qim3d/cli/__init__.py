import argparse
import webbrowser
import platform
import outputformat as ouf
import qim3d
import os

QIM_TITLE = ouf.rainbow(
    rf"""
             _          _____     __ 
      ____ _(_)___ ___ |__  /____/ / 
     / __ `/ / __ `__ \ /_ </ __  /  
    / /_/ / / / / / / /__/ / /_/ /   
    \__, /_/_/ /_/ /_/____/\__,_/    
      /_/                 v{qim3d.__version__}

    """,
    return_str=True,
    cmap="hot",
)


def parse_tuple(arg):
    # Remove parentheses if they are included and split by comma
    return tuple(map(int, arg.strip("()").split(",")))


def main():
    parser = argparse.ArgumentParser(description="qim3d command-line interface.")
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    # GUIs
    gui_parser = subparsers.add_parser("gui", help="Graphical User Interfaces.")

    gui_parser.add_argument(
        "--data-explorer", action="store_true", help="Run data explorer."
    )
    gui_parser.add_argument("--iso3d", action="store_true", help="Run iso3d.")
    gui_parser.add_argument(
        "--annotation-tool", action="store_true", help="Run annotation tool."
    )
    gui_parser.add_argument(
        "--local-thickness", action="store_true", help="Run local thickness tool."
    )
    gui_parser.add_argument(
        "--layers", action="store_true", help="Run Layers."
    )
    gui_parser.add_argument("--host", default="0.0.0.0", help="Desired host.")
    gui_parser.add_argument(
        "--platform", action="store_true", help="Use QIM platform address"
    )
    gui_parser.add_argument(
        "--no-browser", action="store_true", help="Do not launch browser."
    )

    # Viz
    viz_parser = subparsers.add_parser("viz", help="Volumetric visualization.")
    viz_parser.add_argument("source", help="Path to the image file")
    viz_parser.add_argument(
        "-m",
        "--method",
        type=str,
        metavar="METHOD",
        default="itk-vtk",
        help="Which method is used to display file.",
    )
    viz_parser.add_argument(
        "--destination", default="k3d.html", help="Path to save html file."
    )
    viz_parser.add_argument(
        "--no-browser", action="store_true", help="Do not launch browser."
    )

    # Preview
    preview_parser = subparsers.add_parser(
        "preview", help="Preview of the image in CLI"
    )
    preview_parser.add_argument(
        "filename",
        type=str,
        metavar="FILENAME",
        help="Path to image that will be displayed",
    )
    preview_parser.add_argument(
        "--slice",
        type=int,
        metavar="S",
        default=None,
        help="Specifies which slice of the image will be displayed.\nDefaults to middle slice. If number exceeds number of slices, last slice will be displayed.",
    )
    preview_parser.add_argument(
        "--axis",
        type=int,
        metavar="AX",
        default=0,
        help="Specifies from which axis will be the slice taken. Defaults to 0.",
    )
    preview_parser.add_argument(
        "--resolution",
        type=int,
        metavar="RES",
        default=80,
        help="Resolution of displayed image. Defaults to 80.",
    )
    preview_parser.add_argument(
        "--absolute_values",
        action="store_false",
        help="By default set the maximum value to be 255 so the contrast is strong. This turns it off.",
    )

    # File Convert
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert files to different formats without loading the entire file into memory",
    )
    convert_parser.add_argument(
        "input_path",
        type=str,
        metavar="Input path",
        help="Path to image that will be converted",
    )
    convert_parser.add_argument(
        "output_path",
        type=str,
        metavar="Output path",
        help="Path to save converted image",
    )
    convert_parser.add_argument(
        "--chunks",
        type=parse_tuple,
        metavar="Chunk shape",
        default=(64, 64, 64),
        help="Chunk size for the zarr file. Defaults to (64, 64, 64).",
    )
    args = parser.parse_args()

    if args.subcommand == "gui":

        arghost = args.host
        inbrowser = not args.no_browser  # Should automatically open in browser

        interface = None
        if args.data_explorer:
            interface_class = qim3d.gui.data_explorer.Interface
        elif args.iso3d:
            interface_class = qim3d.gui.iso3d.Interface
        elif args.annotation_tool:
            interface_class = qim3d.gui.annotation_tool.Interface
        elif args.local_thickness:
            interface_class = qim3d.gui.local_thickness.Interface
        elif args.layers:
            interface_class = qim3d.gui.layers2d.Interface
        else:
            print(
                "Please select a tool by choosing one of the following flags:\n\t--data-explorer\n\t--iso3d\n\t--annotation-tool\n\t--local-thickness"
            )
            return
        interface = (
            interface_class()
        )  # called here if we add another arguments to initialize

        if args.platform:
            interface.run_interface(host=arghost)
        else:
            interface.launch(inbrowser=inbrowser, force_light_mode=False)

    elif args.subcommand == "viz":

        if args.method == "itk-vtk":

            # We need the full path to the file for the viewer
            current_dir = os.getcwd()
            full_path = os.path.normpath(os.path.join(current_dir, args.source))
            qim3d.viz.itk_vtk(full_path, open_browser = not args.no_browser)


        elif args.method == "k3d":
            volume = qim3d.io.load(str(args.source))
            print("\nGenerating k3d plot...")
            qim3d.viz.volumetric(volume, show=False, save=str(args.destination))
            print(f"Done, plot available at <{args.destination}>")
            if not args.no_browser:
                print("Opening in default browser...")
                webbrowser.open_new_tab(args.destination)
        else:
            raise NotImplementedError(
                f"Method '{args.method}' is not valid. Try 'k3d' or default 'itk-vtk-viewer'"
            )

    elif args.subcommand == "preview":

        image = qim3d.io.load(args.filename)

        qim3d.viz.image_preview(
            image,
            image_width=args.resolution,
            axis=args.axis,
            slice=args.slice,
            relative_intensity=args.absolute_values,
        )

    elif args.subcommand == "convert":

        qim3d.io.convert(args.input_path, args.output_path, chunk_shape=args.chunks)

    elif args.subcommand is None:
        print(QIM_TITLE)
        welcome_text = (
            "\nqim3d is a Python package for 3D image processing and visualization.\n"
            f"For more information, please visit {ouf.c('https://platform.qim.dk/qim3d/', color='orange', return_str=True)}\n"
            " \n"
            "For more information on each subcommand, type 'qim3d <subcommand> --help'.\n"
        )
        print(welcome_text)
        parser.print_help()
        print("\n")


if __name__ == "__main__":
    main()
