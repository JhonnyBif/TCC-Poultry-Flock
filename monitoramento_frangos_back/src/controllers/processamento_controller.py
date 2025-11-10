from flask import Blueprint, request, jsonify
from src.services.processamento_service import ProcessamentoService

processamento_bp = Blueprint('processamento', __name__)
service = ProcessamentoService()

@processamento_bp.route('/processar', methods=['POST'])

def processar():
    try:
        response = service.executar_processamento(request)
        return response, 200
    except Exception as e:
        print(f"Erro ao processar: {e}")
        return jsonify({"erro": str(e)}), 500
