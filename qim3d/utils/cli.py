import argparse
from qim3d.gui import data_explorer, iso3d, annotation_tool, local_thickness, layers2d

def main():
    parser = argparse.ArgumentParser(description='Qim3d command-line interface.')
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

    # subcommands
    gui_parser = subparsers.add_parser('gui', help = 'Graphical User Interfaces.')

    gui_parser.add_argument('--data-explorer', action='store_true', help='Run data explorer.')
    gui_parser.add_argument('--iso3d', action='store_true', help='Run iso3d.')
    gui_parser.add_argument('--annotation-tool', action='store_true', help='Run annotation tool.')
    gui_parser.add_argument('--local-thickness', action='store_true', help='Run local thickness tool.')
    gui_parser.add_argument('--layers2d', action='store_true', help='Run layers2d.')
    gui_parser.add_argument('--host', default='0.0.0.0', help='Desired host.')

    args = parser.parse_args()

    if args.subcommand == 'gui':
        arghost = args.host
        if args.data_explorer:
            
            data_explorer.run_interface(arghost)

        elif args.iso3d:
            iso3d.run_interface(arghost)
        
        elif args.annotation_tool:
            annotation_tool.run_interface(arghost)
        
        elif args.local_thickness:
            local_thickness.run_interface(arghost)

        elif args.layers2d:
            layers2d.run_interface(arghost)
            
if __name__ == '__main__':
    main()