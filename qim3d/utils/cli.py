import argparse
from qim3d.gui import data_explorer, iso3d, annotation_tool, local_thickness

def main():
    parser = argparse.ArgumentParser(description='Qim3d command-line interface.')
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

    # subcommands
    gui_parser = subparsers.add_parser('gui', help = 'Graphical User Interfaces.')

    gui_parser.add_argument('--data-explorer', action='store_true', help='Run data explorer.')
    gui_parser.add_argument('--iso3d', action='store_true', help='Run iso3d.')
    gui_parser.add_argument('--annotation-tool', action='store_true', help='Run annotation tool.')
    gui_parser.add_argument('--local-thickness', action='store_true', help='Run local thickness tool.')
    gui_parser.add_argument('--host', default='0.0.0.0', help='Desired host.')
    gui_parser.add_argument('--platform', action='store_true', help='Use QIM platform address')


    args = parser.parse_args()

    if args.subcommand == 'gui':
        arghost = args.host
        if args.data_explorer:
            if args.platform:
                data_explorer.run_interface(arghost)
            else:
                interface = data_explorer.Interface()
                interface.launch()


        elif args.iso3d:
            if args.platform:
                iso3d.run_interface(arghost)
            else:
                interface = iso3d.Interface()
                interface.launch()            
        
        elif args.annotation_tool:
            if args.platform:
                annotation_tool.run_interface(arghost)
            else:
                interface = annotation_tool.Interface()
                interface.launch()        
        elif args.local_thickness:
            if args.platform:
                local_thickness.run_interface(arghost)
            else:
                interface = local_thickness.Interface()
                interface.launch()
if __name__ == '__main__':
    main()