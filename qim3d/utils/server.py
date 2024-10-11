import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
from qim3d.utils.logger import log

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        """Add CORS headers to each response."""
        # Allow requests from any origin, or restrict to specific domains by specifying the origin
        self.send_header("Access-Control-Allow-Origin", "*")
        # Allow specific methods
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # Allow specific headers (if needed)
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        super().end_headers()

    def list_directory(self, path):
        """Helper to produce a directory listing, includes hidden files."""
        try:
            file_list = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None
        
        # Sort the file list
        file_list.sort(key=lambda a: a.lower())
        
        # Format the list with hidden files included
        displaypath = os.path.basename(path)
        r = ['<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">']
        r.append(f"<html>\n<title>Directory listing for {displaypath}</title>\n")
        r.append(f"<body>\n<h2>Directory listing for {displaypath}</h2>\n")
        r.append("<hr>\n<ul>")
        for name in file_list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            
            # Append the files and directories to the HTML list
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            r.append(f'<li><a href="{linkname}">{displayname}</a></li>')
        r.append("</ul>\n<hr>\n</body>\n</html>\n")
        encoded = "\n".join(r).encode('utf-8', 'surrogateescape')
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        # Write the encoded HTML directly to the response
        self.wfile.write(encoded)

def start_http_server(directory, port=8000):
    """
    Starts an HTTP server serving the specified directory on the given port with CORS enabled.
    
    Parameters:
    directory (str): The directory to serve.
    port (int): The port number to use (default is 8000).
    """
    # Change the working directory to the specified directory
    os.chdir(directory)
    
    # Create the server
    server = HTTPServer(("", port), CustomHTTPRequestHandler)
    
    # Run the server in a separate thread so it doesn't block execution
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    log.info(f"Serving directory '{directory}'\nhttp://localhost:{port}/")
    
    return server
