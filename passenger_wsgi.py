import os
import sys
import traceback

# 1. Ensure current application directory is in sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# 2. Add current running Python version's virtualenv site-packages
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
home_dir = os.path.expanduser("~")
venv_base = os.path.join(home_dir, "virtualenv")
if os.path.exists(venv_base):
    for root, dirs, files in os.walk(venv_base):
        if root.endswith(f"python{py_ver}") and "site-packages" in dirs:
            sp_path = os.path.join(root, "site-packages")
            if sp_path not in sys.path:
                sys.path.insert(0, sp_path)


# 3. Initialize ASGI application via a2wsgi
try:
    from a2wsgi import ASGIMiddleware
    from app import app as fastapi_app
    application = ASGIMiddleware(fastapi_app)
except Exception as e:
    err_msg = traceback.format_exc()
    log_file = os.path.join(app_dir, "passenger_error.log")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n==================== ERROR ON STARTUP ====================\n{err_msg}\n")
    except Exception:
        pass

    def application(environ, start_response):
        status_line = '500 Internal Server Error'
        response_headers = [
            ('Content-Type', 'text/html; charset=utf-8'),
            ('Access-Control-Allow-Origin', '*')
        ]
        start_response(status_line, response_headers)
        html_output = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>500 Application Startup Error</title>
            <style>
                body {{ font-family: monospace; padding: 20px; background: #1a1a1a; color: #ff6b6b; }}
                h2 {{ color: #f06595; }}
                pre {{ background: #2d2d2d; padding: 15px; border-radius: 8px; color: #f8f9fa; overflow-x: auto; white-space: pre-wrap; }}
                .hint {{ color: #74c0fc; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <h2>⚠️ cPanel Python Application Startup Error</h2>
            <p>The Python application failed to initialize. Review the traceback below:</p>
            <pre>{err_msg}</pre>
            <div class="hint">
                <p><strong>Common Solutions:</strong></p>
                <ul>
                    <li>Run <code>pip install -r requirements.txt</code> in your cPanel Python App terminal.</li>
                    <li>Ensure <code>opencv-python-headless</code> is installed (NOT <code>opencv-python</code>).</li>
                    <li>Check that <code>passenger_wsgi.py</code> and <code>app.py</code> are in the Application Root.</li>
                </ul>
            </div>
        </body>
        </html>
        """
        return [html_output.encode('utf-8')]

