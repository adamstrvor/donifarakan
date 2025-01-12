#-----------------------------------------
# LIBRARIES
#-----------------------------------------

from flask import Flask, jsonify
from .routes import api_bp
from .config import *

#-----------------------------------------
# INIT
#-----------------------------------------

def start (ip_address="127.0.0.1",port=5000):
    app = Flask(__name__)

    app.register_blueprint(api_bp)


    #-----------------------------------------
    # ERROR HANDLING
    #-----------------------------------------

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Not Found", "error_message": str(error), 'lang':supported_languages}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal Server Error", "error_message": "An unexpected error occurred.",'lang':supported_languages}), 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        # You can also log the exception here
        return jsonify({"error": "Internal Server Error", "error_message": str(e),'lang':supported_languages}), 500


    app.run(host=ip_address, port=port, debug=True)
