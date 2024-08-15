from flask import Flask
from .routes import main as main_blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main_blueprint)
    return app

from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    # Print template search path for debugging
    print(f"Template search path: {app.template_folder}")
    print(f"Absolute path: {os.path.abspath(app.template_folder)}")

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
