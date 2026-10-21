"""DSU Admission Cell Management Portal - Frontend Server.

Serves Website 2 on http://127.0.0.1:5501
- Staff Login: http://127.0.0.1:5501/ or http://127.0.0.1:5501/login
- Protected Dashboard: http://127.0.0.1:5501/dashboard
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
PORT = 5501

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")


class AdmissionFrontendHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for DSU Admission Cell portal website."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        # Strip query parameters and fragments for path resolution
        path = self.path.split("?")[0].split("#")[0]

        # Login routes: /, /login, /admin/login, /admin-login.html
        if path in ("/", "/login", "/admin/login", "/admin-login.html"):
            target_file = STATIC_DIR / "admin-login.html"
            return self._serve_file(target_file, "text/html; charset=utf-8")

        # Dashboard routes: /dashboard, /admin/dashboard, /admin-dashboard.html
        if path in ("/dashboard", "/admin/dashboard", "/admin-dashboard.html"):
            target_file = STATIC_DIR / "admin-dashboard.html"
            return self._serve_file(target_file, "text/html; charset=utf-8")

        # Support direct /static/..., /css/..., and /js/... paths
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            target_file = STATIC_DIR / rel_path
            if target_file.is_file():
                return self._serve_file(target_file)
        elif path.startswith("/css/") or path.startswith("/js/"):
            target_file = STATIC_DIR / path.lstrip("/")
            if target_file.is_file():
                return self._serve_file(target_file)

        # Fallback to standard static file serving from BASE_DIR or STATIC_DIR
        file_candidate = BASE_DIR / path.lstrip("/")
        if file_candidate.is_file():
            return self._serve_file(file_candidate)

        static_candidate = STATIC_DIR / path.lstrip("/")
        if static_candidate.is_file():
            return self._serve_file(static_candidate)

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
        sys.stderr.write(f"[DSU-Admission-5501] {self.address_string()} - {format % args}\n")


def run():
    server_address = (HOST, PORT)
    httpd = http.server.ThreadingHTTPServer(server_address, AdmissionFrontendHandler)
    print("=" * 70)
    print(f" DSU Admission Cell Management Portal")
    print(f" Staff Login: http://{HOST}:{PORT}/login (or http://{HOST}:{PORT}/)")
    print(f" Dashboard:   http://{HOST}:{PORT}/dashboard")
    print(f" Shared Backend: http://127.0.0.1:8000")
    print(" Press Ctrl+C to terminate.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nAdmission Cell frontend server stopping...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
