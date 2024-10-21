from pathlib import Path
import subprocess
import os
import platform

from .helpers import get_itk_dir, get_nvm_dir, get_node_binaries_dir, get_viewer_dir, SOURCE_FNM, NotInstalledError, run_for_platform



class Installer:
    """
    Implements installation procedure of itk-vtk-viewer for each OS. 
    Also goes for minimal installation: checking if the necessary binaries aren't already installed
    """
    def __init__(self):
        self.platform = platform.system()
        self.install_functions = (self.install_node_manager, self.install_node, self.install_viewer)

        self.dir = get_itk_dir() # itk_vtk_viewer folder within qim3d.viz

        # If nvm was already installed, there should be this environment variable
        # However it could have also been installed via our process, or user deleted the folder but didn't adjusted the bashrc, that's why we check again
        self.os_nvm_dir = os.getenv('NVM_DIR')
        if self.os_nvm_dir is not None:
            self.os_nvm_dir = Path(self.os_nvm_dir)
        self.qim_nvm_dir = get_nvm_dir(self.dir)
        if not os.path.isdir(self.qim_nvm_dir):
            os.mkdir(self.qim_nvm_dir)

    @property
    def is_node_manager_already_installed(self) -> bool:
        """
        Checks for global and local installation of nvm (Node Version Manager)
        """
        def _linux() -> bool:
            command_f = lambda nvmsh: F'/bin/bash -c "source {nvmsh} && nvm"'
            if self.os_nvm_dir is not None:
                nvmsh = self.os_nvm_dir.joinpath('nvm.sh')
                output = subprocess.run(command_f(nvmsh), shell = True, capture_output = True)
                if not output.stderr:
                    self.nvm_dir = self.os_nvm_dir
                    return True
                
            nvmsh = self.qim_nvm_dir.joinpath('nvm.sh')
            output = subprocess.run(command_f(nvmsh), shell = True, capture_output = True)
            self.nvm_dir = self.qim_nvm_dir
            return not bool(output.stderr) # If there is an error running the above command then it is not installed (not in expected location)
        
        def _windows() -> bool:
            output = subprocess.run(['powershell.exe', 'fnm --version'], capture_output=True)
            return not bool(output.stderr)
        
        return run_for_platform(linux_func=_linux, windows_func=_windows,macos_func= _linux)

    @property
    def is_node_already_installed(self) -> bool:
        """
        Checks for global and local installation of Node.js and npm (Node Package Manager)
        """
        def _linux() -> bool:
            # get_node_binaries_dir might return None if the folder is not there
            # In that case there is 'None' added to the PATH, thats not a problem
            # the command will return an error to the output and it will be evaluated as not installed
            command = F'export PATH="$PATH:{get_node_binaries_dir(self.nvm_dir)}" && npm version'

            output = subprocess.run(command, shell = True, capture_output = True)
            return not bool(output.stderr)
        
        def _windows() -> bool:
            # Didn't figure out how to install the viewer and run it properly when using global npm
            return False
            
        return run_for_platform(linux_func=_linux,windows_func= _windows,macos_func= _linux)

        

    def install(self):
        """
        First check if some of the binaries are not installed already. 
        If node.js is already installed (it was able to call npm without raising an error)
            it only has to install the viewer and doesn't have to go through the process
        """
        if self.is_node_manager_already_installed:
            self.install_status = 1
            print("Node manager already installed")
            if self.is_node_already_installed:
                self.install_status = 2
                print("Node.js already installed")
            
        else:
            self.install_status = 0
        
        for install_function in self.install_functions[self.install_status:]:
            install_function()

    def install_node_manager(self):
        def _linux():
            print(F'Installing Node manager into {self.nvm_dir}...')
            _ = subprocess.run([F'export NVM_DIR={self.nvm_dir} && bash {self.dir.joinpath("install_nvm.sh")}'], shell = True, capture_output=True)

        def _windows():
            print("Installing node manager...")
            subprocess.run(["powershell.exe", F'$env:XDG_DATA_HOME = "{self.dir}";', "winget install Schniz.fnm"])


        # self._run_for_platform(_linux, None, _windows)
        run_for_platform(linux_func=_linux,windows_func= _windows,macos_func= _linux)
        print("Node manager installed")

    def install_node(self):
        def _linux():
            """
            If nvm was already installed, terminal should have environemnt variable 'NVM_DIR' where is nvm.sh
            We have to source that file either way, to be able to call nvm function
            If it was't installed before, we need to export NVM_DIR in order to install npm to correct location
            """
            print(F'Installing node.js into {self.nvm_dir}...')
            if self.install_status == 0:
                nvm_dir = self.nvm_dir
                prefix = F'export NVM_DIR={nvm_dir} && '

            elif self.install_status == 1:
                nvm_dir = self.os_nvm_dir
                prefix = ''
                
            nvmsh = Path(nvm_dir).joinpath('nvm.sh')
            command = f'{prefix}/bin/bash -c "source {nvmsh} && nvm install 22"'
            output = subprocess.run(command, shell = True, capture_output=True)

        def _windows():
            subprocess.run(["powershell.exe",F'$env:XDG_DATA_HOME = "{self.dir}";', SOURCE_FNM, F"fnm use --fnm-dir {self.dir} --install-if-missing 22"])
            
        print(F'Installing node.js...')
        run_for_platform(linux_func = _linux, windows_func=_windows, macos_func=_linux)
        print("Node.js installed")

    def install_viewer(self):
        def _linux():
            # Adds local binaries to the path in case we had to install node first (locally into qim folder), but shouldnt interfere even if 
            # npm is installed globally
            command = F'export PATH="$PATH:{get_node_binaries_dir(self.nvm_dir)}" && npm install --prefix {self.viewer_dir} itk-vtk-viewer'
            output = subprocess.run([command], shell=True, capture_output=True)
            # print(output.stderr)

        def _windows():
            try:
                node_bin = get_node_binaries_dir(self.dir)
                print(F'Installing into {self.viewer_dir}')
                subprocess.run(["powershell.exe", F'$env:PATH=$env:PATH + \';{node_bin}\';', F"npm install --prefix {self.viewer_dir} itk-vtk-viewer"], capture_output=True)
            except NotInstalledError: # Not installed in qim
                subprocess.run(["powershell.exe", SOURCE_FNM, F"npm install itk-vtk-viewer"], capture_output=True)
            

        self.viewer_dir = get_viewer_dir(self.dir)
        if not os.path.isdir(self.viewer_dir):
            os.mkdir(self.viewer_dir)

        print(F"Installing itk-vtk-viewer...")
        run_for_platform(linux_func=_linux, windows_func=_windows, macos_func=_linux)
        print("Itk-vtk-viewer installed")

    