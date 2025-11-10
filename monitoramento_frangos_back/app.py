from flask import Flask, request, jsonify
from flask_cors import CORS


# Factory da aplicação
def create_app():
    app = Flask(__name__)
    CORS(app)

    from src.controllers.processamento_controller import processamento_bp
    app.register_blueprint(processamento_bp)


    return app

# Execução direta
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
