import os
import cv2
import numpy as np
import tempfile
from collections import defaultdict
from ultralytics import YOLO, settings

from src.utils.carregar_nome_classes import carregar_nomes_classes
from src.utils.calcular_homografia import calcular_homografia
from src.utils.converter_video_ffmpeg import converter_video_ffmpeg
from src.utils.gerar_heatmap_frame import gerar_heatmap_frame
from src.utils.calcular_metricas_finais import calcular_metricas_finais
from src.utils.desenhar_malha import desenhar_malha


def detectar_video_memoria_metro_quadrado(video_bytes, superior, esquerdo, direito, inferior, limite_aglomeracao):
    print(limite_aglomeracao, 'Liemte de aglomeração recebido na função')
    with tempfile.TemporaryDirectory() as tempdir:
        temp_video_path = os.path.join(tempdir, "input.mp4")
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_video_path)
        largura_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 10

        settings.update({
            'weights_dir': '../weights',
            'datasets_dir': '../datasets',
            'runs_dir': '../runs'
        })

        nomes_classes = carregar_nomes_classes('custom_dataset')
        weights_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'train', 'weights', 'best.pt'))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Modelo não encontrado: {weights_path}")

        model = YOLO(weights_path)

        pontos_imagem = np.float32([
            [0, 0],
            [largura_img - 1, 0],
            [largura_img - 1, altura_img - 1],
            [0, altura_img - 1]
        ])
        pontos_reais = np.float32([
            [0, 0],
            [superior, 0],
            [inferior, direito],
            [0, esquerdo]
        ])

        largura_m = max(superior, inferior)
        altura_m = max(esquerdo, direito)
        escala = 300
        largura_real_px = int(largura_m * escala)
        altura_real_px = int(altura_m * escala)
        passo = escala

        H = calcular_homografia(pontos_imagem, pontos_reais, escala)
        H_inv = np.linalg.inv(H)

        contagem_por_quadrado = defaultdict(int)
        contador_aglomeracao = defaultdict(int)
        total_deteccoes = defaultdict(int)
        frames_por_classe = defaultdict(set)
        todos_centros = []
        frame_atual = 0

        path_boxes = os.path.join(tempdir, "boxes.mp4")
        path_heatmap = os.path.join(tempdir, "heatmap.mp4")

        writer_boxes = cv2.VideoWriter(path_boxes, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura_img, altura_img))
        writer_heatmap = cv2.VideoWriter(path_heatmap, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura_img, altura_img))

        results_stream = model.track(source=temp_video_path, stream=True, conf=0.70, iou=0.7,  max_det=1500,)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_atual += 1
            result = next(results_stream)
            frame_verde = frame.copy()

            if result.boxes and result.boxes.xyxy.shape[0] > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes_detectadas = list(map(int, result.boxes.cls.cpu().tolist()))

                for i, classe_id in enumerate(classes_detectadas):
                    nome_classe = nomes_classes[classe_id] if 0 <= classe_id < len(nomes_classes) else f"Classe {classe_id}"
                    total_deteccoes[nome_classe] += 1
                    frames_por_classe[nome_classe].add(frame_atual)

                    x1, y1, x2, y2 = boxes[i]
                    centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                    centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]

                    if frame_atual >= 3:
                        todos_centros.append(centro_corrigido.tolist())

                    cx, cy = centro_corrigido.astype(int)
                    if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
                        col = cx // passo
                        lin = cy // passo
                        chave = (lin, col)
                        contagem_por_quadrado[chave] += 1

                    cv2.circle(frame_verde, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame_verde, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            desenhar_malha(frame_verde, H_inv, contagem_por_quadrado, frame_atual, passo, altura_real_px, largura_real_px, limite_aglomeracao, contador_aglomeracao)

            heatmap = gerar_heatmap_frame(frame, H_inv, contagem_por_quadrado, frame_atual, passo, altura_real_px, largura_real_px, limite_aglomeracao, contador_aglomeracao)
            blend = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            writer_boxes.write(frame_verde)
            writer_heatmap.write(blend)

        cap.release()
        writer_boxes.release()
        writer_heatmap.release()

        converted_boxes = converter_video_ffmpeg(path_boxes, ".mp4")
        converted_heatmap = converter_video_ffmpeg(path_heatmap, ".mp4")

        with open(converted_boxes, 'rb') as f:
            video_boxes = f.read()
        with open(converted_heatmap, 'rb') as f:
            video_heatmap = f.read()

        area_total_m2, area_detectada_m2 = calcular_metricas_finais(pontos_reais, todos_centros, frame_atual, altura_real_px, largura_real_px)
        total_frangos = sum(total_deteccoes.values())
        media_frangos_por_frame = total_frangos / frame_atual if frame_atual > 0 else 0
        densidade_total_imagem = media_frangos_por_frame / area_total_m2
        densidade_apenas_detectado = media_frangos_por_frame / area_detectada_m2 if area_detectada_m2 > 0 else 0

        output_text = ""
        for nome_classe, total in total_deteccoes.items():
            num_frames = len(frames_por_classe.get(nome_classe, []))
            media = total / num_frames if num_frames > 0 else 0
            output_text += f"{nome_classe}: {total} detecções em {num_frames} frames -> Média: {media:.2f} detecções/frame\n"

        output_text += f"\n--- MÉTRICAS DE DENSIDADE ---\n"
        output_text += f"Média de frangos por frame: {media_frangos_por_frame:.2f}\n"
        output_text += f"Área total da imagem (real): {area_total_m2:.2f} m²\n"
        output_text += f"Área com frangos detectados (convexa): {area_detectada_m2:.2f} m²\n"
        output_text += f"Densidade total da imagem: {densidade_total_imagem:.2f} frangos/m²\n"
        output_text += f"Densidade apenas onde há frangos: {densidade_apenas_detectado:.2f} frangos/m²\n"

        return video_boxes, video_heatmap, output_text
