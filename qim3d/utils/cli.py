import argparse
import webbrowser

from qim3d.gui import annotation_tool, data_explorer, iso3d, local_thickness
from qim3d.io.loading import DataLoader
from qim3d.utils import image_preview
from qim3d import __version__ as version
import outputformat as ouf
import qim3d

QIM_TITLE = ouf.rainbow(
    f"\n         _          _____     __ \n  ____ _(_)___ ___ |__  /____/ / \n / __ `/ / __ `__ \ /_ </ __  /  \n/ /_/ / / / / / / /__/ / /_/ /   \n\__, /_/_/ /_/ /_/____/\__,_/    \n  /_/                 v{version}\n\n",
    return_str=True,
    cmap="hot",
)


def main():
    parser = argparse.ArgumentParser(description="Qim3d command-line interface.")
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
    gui_parser.add_argument("--host", default="0.0.0.0", help="Desired host.")
    gui_parser.add_argument(
        "--platform", action="store_true", help="Use QIM platform address"
    )
    gui_parser.add_argument(
        "--no-browser", action="store_true", help="Do not launch browser."
    )

    # K3D
    viz_parser = subparsers.add_parser("viz", help="Volumetric visualization.")
    viz_parser.add_argument("--source", default=False, help="Path to the image file")
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

    args = parser.parse_args()

    if args.subcommand == "gui":
        arghost = args.host
        inbrowser = not args.no_browser  # Should automatically open in browser
        if args.data_explorer:
            if args.platform:
                data_explorer.run_interface(arghost)
            else:
                interface = data_explorer.Interface()
                interface.launch(inbrowser=inbrowser)
        elif args.iso3d:
            if args.platform:
                iso3d.run_interface(arghost)
            else:
                interface = iso3d.Interface()
                interface.launch(inbrowser=inbrowser)

        elif args.annotation_tool:
            if args.platform:
                annotation_tool.run_interface(arghost)
            else:
                interface = annotation_tool.Interface()
                interface.launch(inbrowser=inbrowser)
        elif args.local_thickness:
            if args.platform:
                local_thickness.run_interface(arghost)
            else:
                interface = local_thickness.Interface()
                interface.launch(inbrowser=inbrowser)
    elif args.subcommand == "viz":
        if not args.source:
            print("Please specify a source file using the argument --source")
            return
        # Load the data
        print(f"Loading data from {args.source}")
        volume = qim3d.io.load(str(args.source))
        print(f"Done, volume shape: {volume.shape}")

        # Make k3d plot
        print("\nGenerating k3d plot...")
        qim3d.viz.vol(volume, show=False, save=str(args.destination))
        print(f"Done, plot available at <{args.destination}>")

        if not args.no_browser:
            print("Opening in default browser...")
            webbrowser.open_new_tab(args.destination)

    elif args.subcommand == "preview":
        image = DataLoader().load(args.filename)
        image_preview(
            image,
            image_width=args.resolution,
            axis=args.axis,
            slice=args.slice,
            relative_intensity=args.absolute_values,
        )

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
