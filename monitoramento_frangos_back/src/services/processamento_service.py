import os
import json
import base64
import tempfile
import re
from datetime import datetime
from flask import jsonify
from werkzeug.utils import secure_filename
from src.yolo.processamento_video.gerar_video_superior import processar_video_memoria
from src.yolo.deteccao.predict_quantidade_metro_quadrado import detectar_video_memoria_metro_quadrado
from src.yolo.deteccao.predict_img_metro_quadrado import detectar_video_image_memoria_metro_quadrado, detectar_image_memoria_metro_quadrado


class ProcessamentoService:

    def executar_processamento(self, request):
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "Arquivo de vídeo não enviado"}), 400

        datasets = self._carregar_datasets(request)
        if datasets is None:
            return jsonify({"error": "Formato inválido em datasets"}), 400

        selected_option = request.form.get("selectedOption")
        parte_cima = request.form.get("parteCima") == "true"
        parte_baixo = request.form.get("parteBaixo") == "true"
        temp_input_path = self._salvar_temporariamente(file)
            
        resultados = {}
       
        if "Detectar Quantidade Por Metro Quadrado" in datasets:
            resultados["quantidade_por_metro_quadrado"] = self._processar_metro_quadrado(
                file=file,
                path=temp_input_path,
                request=request,
                tipo=selected_option,
                cima=parte_cima,
                baixo=parte_baixo
            ).json
        return jsonify(resultados)

    def extrair_densidade_apenas_com_frangos(self, texto_resultado: str) -> float:
        print(f"Texto de resultado recebido: {texto_resultado}")
        for linha in texto_resultado.splitlines():
            if "Densidade apenas onde há frangos" in linha:
                match = re.search(r'([\d.]+)', linha)
                if match:
                    return float(match.group(1))
        raise ValueError("Linha com densidade não encontrada ou mal formatada.")

    def _salvar_temporariamente(self, file):
        filename = secure_filename(file.filename)
        path = os.path.join(tempfile.gettempdir(), filename)
        file.save(path)
        return path

    def _carregar_datasets(self, request):
        raw = request.form.get("datasets")
        try:
            return json.loads(raw) if raw else []
        except json.JSONDecodeError:
            return None

    def _converter_data(self, data_str):
        return datetime.strptime(data_str.split(" GMT")[0], "%a %b %d %Y %H:%M:%S")

    def _calcular_idade(self, request):
        idade = request.form.get("idade")
        if idade and idade != "null":
            return int(idade)
        data_entrada = self._converter_data(request.form.get("dataEntrada"))
        data_video = self._converter_data(request.form.get("dataVideo"))
        return (data_video - data_entrada).days

    def _definir_limite_aglomeracao(self, idade):
        print(f"Calculando limite de aglomeração para idade: {idade}")
        if idade <= 27: return 18
        if idade <= 30: return 15
        if idade <= 34: return 12
        if idade <= 40: return 10
        if idade <= 47: return 8
        return 6

    def _ler_bytes_video(self, path):
        with open(path, 'rb') as f:
            return f.read()

    def _encode_base64(self, data):
        return base64.b64encode(data).decode('utf-8')

    def _processar_metro_quadrado(self, file, path, request, tipo, cima, baixo):
        idade = self._calcular_idade(request)
        limite = self._definir_limite_aglomeracao(idade)
        print(f"Limite de aglomeração definido: {limite}")
        sup = float(request.form.get("superior", 0))
        esq = float(request.form.get("esquerda", 0))
        dir = float(request.form.get("direita", 0))
        inf = float(request.form.get("baixo", 0))

        if tipo == 'video':
            video_bytes = processar_video_memoria(path, cima, baixo, False) if (cima or baixo) else self._ler_bytes_video(path)
            video_proc, heatmap, resultado = detectar_video_memoria_metro_quadrado(video_bytes, sup, esq, dir, inf, limite)
            return jsonify({
                "resultado": resultado,
                "video_base64": self._encode_base64(video_proc),
                "heatmap": self._encode_base64(heatmap)
            })

        if tipo == 'fragment':
            video_bytes = processar_video_memoria(path, cima, baixo, True) if (cima or baixo) else self._ler_bytes_video(path)
            imgs, heatmaps, resultado = detectar_video_image_memoria_metro_quadrado(video_bytes, sup, esq, dir, inf, limite)
            return jsonify({
                "resultado": resultado,
                "imagens_base64": imgs,
                "imagens_heatmap_base64": heatmaps
            })

        if tipo == 'image':
            import time
            start_time = time.time()
            img, heat, resultado = detectar_image_memoria_metro_quadrado(file, sup, esq, dir, inf, limite)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Tempo de execução: {execution_time} segundos")
            return jsonify({
                "resultado": resultado,
                "image_base64": self._encode_base64(img),
                "image_heatmap_base64": self._encode_base64(heat)
            })