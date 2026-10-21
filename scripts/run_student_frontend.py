"""DSU College Enquiry Chatbot - Student Website Frontend Server.

Serves Website 1 on http://127.0.0.1:5500
- Root (/) serves static/index.html with the embedded floating chatbot widget.
- Serves static assets (/static/css, /static/js, etc.).
"""

import http.server
import mimetypes
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
HOST = "127.0.0.1"
PORT = 5500

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")


class StudentFrontendHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for DSU Student Enquiry Chatbot website."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        # Strip query parameters and fragments for path resolution
        path = self.path.split("?")[0].split("#")[0]

        # Route root or index to static/index.html
        if path in ("/", "/index.html"):
            target_file = STATIC_DIR / "index.html"
            return self._serve_file(target_file, "text/html; charset=utf-8")

        # Support direct /css/... and /js/... paths as well as /static/...
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            target_file = STATIC_DIR / rel_path
            if target_file.is_file():
                return self._serve_file(target_file)
        elif path.startswith("/css/") or path.startswith("/js/"):
            target_file = STATIC_DIR / path.lstrip("/")
            if target_file.is_file():
                return self._serve_file(target_file)

        # Fallback to standard static file serving
        file_candidate = BASE_DIR / path.lstrip("/")
        if file_candidate.is_file():
            return self._serve_file(file_candidate)

        # Default 404
        self.send_error(404, f"File not found: {path}")

    def _serve_file(self, file_path: Path, forced_content_type: str = None):
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            content_type = forced_content_type
            if not content_type:
                ctype, _ = mimetypes.guess_type(str(file_path))
                content_type = ctype or "application/octet-stream"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[DSU-Student-5500] {self.address_string()} - {format % args}\n")


def run():
    server_address = (HOST, PORT)
    httpd = http.server.ThreadingHTTPServer(server_address, StudentFrontendHandler)
    print("=" * 70)
    print(f" DSU College Enquiry Chatbot (Student Website)")
    print(f" URL: http://{HOST}:{PORT}")
    print(f" Shared Backend: http://127.0.0.1:8000")
    print(" Press Ctrl+C to terminate.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStudent frontend server stopping...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
