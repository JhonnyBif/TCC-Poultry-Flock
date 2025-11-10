import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
import base64
from collections import defaultdict

from src.utils.carregar_nome_classes import carregar_nomes_classes
from src.utils.calcular_homografia import calcular_homografia
from src.utils.converter_video_ffmpeg import converter_video_ffmpeg
from src.utils.gerar_heatmap_frame import gerar_heatmap_frame
from src.utils.calcular_metricas_finais import calcular_metricas_finais
from src.utils.desenhar_malha import desenhar_malha_img


import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
nomes_classes = carregar_nomes_classes('custom_dataset')
weights_path = os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', 'train', 'weights', 'best.pt'))
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Modelo não encontrado: {weights_path}")
model = YOLO(weights_path)

def detectar_video_image_memoria_metro_quadrado(video_bytes, superior, esquerdo, direito, inferior, limite_aglomeracao):
   with tempfile.TemporaryDirectory() as tempdir:
        temp_video_path = os.path.join(tempdir, "input.mp4")
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)
    
        cap = cv2.VideoCapture(temp_video_path)
        largura_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = 5 * fps
        frame_count = 0

        imagens_base64_boxes = []
        imagens_base64_heatmap = []
        total_deteccoes = defaultdict(int)
        frames_por_classe = defaultdict(set)
        todos_centros = []
        contagem_por_quadrado = {}

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
        escala = 300
        largura_m = max(superior, inferior)
        altura_m = max(esquerdo, direito)
        largura_real_px = int(largura_m * escala)
        altura_real_px = int(altura_m * escala)
        passo = int(1.0 * escala)

        H = calcular_homografia(pontos_imagem, pontos_reais, escala)
        H_inv = np.linalg.inv(H)
        heatmap_acumulado = np.zeros((altura_real_px, largura_real_px), dtype=np.float32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = model.track(
                    source=frame,
                    conf=0.40,
                    iou=0.7,
                    imgsz=640,
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    max_det=1500,
                    line_width=1
                )
                resultado = results[0]
                boxes = resultado.boxes.xyxy.cpu().numpy() if resultado.boxes is not None else []
                classes_detectadas = resultado.boxes.cls.tolist() if resultado.boxes is not None else []

                frame_boxes = frame.copy()
                for i, classe_id in enumerate(classes_detectadas):
                    classe_id = int(classe_id)
                    nome_classe = nomes_classes[classe_id] if 0 <= classe_id < len(nomes_classes) else f"Classe {classe_id}"

                    total_deteccoes[nome_classe] = total_deteccoes.get(nome_classe, 0) + 1
                    frames_por_classe.setdefault(nome_classe, set()).add(frame_count)

                    x1, y1, x2, y2 = boxes[i]
                    centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                    centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
                    todos_centros.append(centro_corrigido.tolist())

                    cx, cy = centro_corrigido.astype(int)
                    if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
                        heatmap_acumulado[cy, cx] += 1
                        col = int(cx // passo)
                        lin = int(cy // passo)
                        chave = (lin, col)
                        contagem_por_quadrado[chave] = contagem_por_quadrado.get(chave, 0) + 1

                    cv2.circle(frame_boxes, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

                desenhar_malha_img(frame_boxes, H_inv, contagem_por_quadrado, frame_count, frame_interval, passo, altura_real_px, largura_real_px, limite_aglomeracao)
                _, buffer_boxes = cv2.imencode('.jpg', frame_boxes)
                imagens_base64_boxes.append(base64.b64encode(buffer_boxes).decode('utf-8'))

                heatmap_frame = np.zeros_like(frame)
                for lin in range(0, altura_real_px, passo):
                    for col in range(0, largura_real_px, passo):
                        quad_corrigido = np.array([[ [col, lin], [col + passo, lin], [col + passo, lin + passo], [col, lin + passo] ]], dtype=np.float32)
                        quad_original = cv2.perspectiveTransform(quad_corrigido, H_inv).astype(int)
                        chave = (lin // passo, col // passo)
                        if chave in contagem_por_quadrado:
                            media_frame = contagem_por_quadrado[chave] / (frame_count // frame_interval + 1)
                            if media_frame > limite_aglomeracao:
                                cv2.fillPoly(heatmap_frame, [quad_original], (0, 0, 255))
                            else:
                                cor = int(min(255, media_frame * 25))
                                cv2.fillPoly(heatmap_frame, [quad_original], (0, cor, 255 - cor))
                blend = cv2.addWeighted(frame, 0.5, heatmap_frame, 0.5, 0)
                _, buffer_heatmap = cv2.imencode('.jpg', blend)
                imagens_base64_heatmap.append(base64.b64encode(buffer_heatmap).decode('utf-8'))

            frame_count += 1

        cap.release()
        os.remove(temp_video_path)

        area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
        total_frangos = sum(total_deteccoes.values())
        num_frames_utilizados = frame_count // frame_interval if frame_interval > 0 else 1
        media_frangos_por_frame = total_frangos / num_frames_utilizados if num_frames_utilizados > 0 else 0
        densidade_total_imagem = media_frangos_por_frame / area_total_m2 if area_total_m2 > 0 else 0

        if len(todos_centros) >= 3:
            pontos = np.array(todos_centros).astype(np.int32)
            hull = cv2.convexHull(pontos)
            mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            area_detectada_px = np.count_nonzero(mask)
            m2_por_px2 = area_total_m2 / (altura_real_px * largura_real_px)
            area_detectada_m2 = area_detectada_px * m2_por_px2
            densidade_apenas_detectado = media_frangos_por_frame / area_detectada_m2 if area_detectada_m2 > 0 else 0
        else:
            area_detectada_m2 = 0
            densidade_apenas_detectado = 0

        resultado_txt = ""
        for nome_classe, total in total_deteccoes.items():
            num_frames = len(frames_por_classe.get(nome_classe, []))
            media = total / num_frames if num_frames > 0 else 0
            resultado_txt += f"{nome_classe}: {total} detecções em {num_frames} frames -> Média: {media:.2f} detecções/frame\n"

        resultado_txt += "\n--- MÉTRICAS DE DENSIDADE ---\n"
        resultado_txt += f"Média de frangos por frame: {media_frangos_por_frame:.2f}\n"
        resultado_txt += f"Área total da imagem (real): {area_total_m2:.2f} m²\n"
        resultado_txt += f"Área com frangos detectados (convexa): {area_detectada_m2:.2f} m²\n"
        resultado_txt += f"Densidade total da imagem: {densidade_total_imagem:.2f} frangos/m²\n"
        resultado_txt += f"Densidade apenas onde há frangos: {densidade_apenas_detectado:.2f} frangos/m²\n"

        return imagens_base64_boxes, imagens_base64_heatmap, resultado_txt


# ORIGINAL

def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_aglomeracao):
    print(limite_aglomeracao, 'Liemte de aglomeração recebido na função')
    file_storage.stream.seek(0)
    file_bytes = file_storage.read()
    if not file_bytes:
        raise ValueError("Arquivo vazio ou não pôde ser lido.")

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Falha ao decodificar imagem.")

    # --- Homografia ---
    altura, largura, _ = img.shape
    pontos_imagem = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])
    pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])
    escala = 300
    largura_m = max(superior, inferior)
    altura_m = max(esquerdo, direito)
    largura_real_px = int(largura_m * escala)
    altura_real_px = int(altura_m * escala)
    passo = int(1.0 * escala)

    H = calcular_homografia(pontos_imagem, pontos_reais, escala)
    H_inv = np.linalg.inv(H)

    # --- Predição YOLO ---
    results = model.track(
        source=img,
        conf=0.7,
        iou=0.7,
        imgsz=1280,
        save=False,
        save_txt=False,
        save_conf=False,
        max_det=1500,
        line_width=1
    )

    resultado = results[0]
    total_deteccoes = {}
    todos_centros = []
    contagem_por_quadrado = {}

    frame_boxes = img.copy()
    heatmap_frame = np.zeros_like(img)

    if resultado.boxes is not None:
        boxes = resultado.boxes.xyxy.cpu().numpy()
        classes_detectadas = resultado.boxes.cls.tolist()

        for i, classe_id in enumerate(classes_detectadas):
            classe_id = int(classe_id)
            nome_classe = nomes_classes[classe_id] if 0 <= classe_id < len(nomes_classes) else f"Classe {classe_id}"
            total_deteccoes[nome_classe] = total_deteccoes.get(nome_classe, 0) + 1

            x1, y1, x2, y2 = boxes[i]
            centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
            centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
            todos_centros.append(centro_corrigido.tolist())

            cx, cy = centro_corrigido.astype(int)
            if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
                col = int(cx // passo)
                lin = int(cy // passo)
                chave = (lin, col)
                contagem_por_quadrado[chave] = contagem_por_quadrado.get(chave, 0) + 1

            # Desenha bounding box e centro
            cv2.circle(frame_boxes, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
            cv2.rectangle(frame_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    # --- Desenha Malha e Prepara Heatmap ---
    for lin in range(0, altura_real_px, passo):
        for col in range(0, largura_real_px, passo):
            quad_corrigido = np.array([[ 
                [col, lin], [col + passo, lin], [col + passo, lin + passo], [col, lin + passo]
            ]], dtype=np.float32)
            quad_original = cv2.perspectiveTransform(quad_corrigido, H_inv).astype(int)

            # Malha na imagem de boxes
            cv2.polylines(frame_boxes, [quad_original], isClosed=True, color=(0, 255, 0), thickness=4)

            # Dados e média para cada quadrado
            chave = (lin // passo, col // passo)
            if chave in contagem_por_quadrado:
                media = contagem_por_quadrado[chave]
                pos_texto = tuple(quad_original[0][0])
                texto = f"{media:.1f}"
                if media > limite_aglomeracao:
                    texto += " AGLOMERACAO"
                cv2.putText(frame_boxes, texto, (pos_texto[0]+1, pos_texto[1]+11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 8)
                cv2.putText(frame_boxes, texto, (pos_texto[0], pos_texto[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print(f"Quadrado {chave}: {media} detecções")
                print(f"Quadrado {chave}: {media} detecções")
                print(f"Quadrado {chave}: {media} detecções")
                print(f"Quadrado {chave}: {media} detecções")
                # Cor no heatmap
                if media > limite_aglomeracao:
                    cv2.fillPoly(heatmap_frame, [quad_original], (0, 0, 255))
                else:
                    porcentagem = min(1.0, media / limite_aglomeracao)
                    if porcentagem <= 0.60:
                        cor = (0, 255, 0)
                    else:
                        esc = (porcentagem - 0.5) / 0.5
                        cor = (0, int((1.0 - esc) * 2 * 225), 255)
                    cv2.fillPoly(heatmap_frame, [quad_original], cor)

    # --- Geração imagem heatmap sobreposta ---
    heatmap_blend = cv2.addWeighted(img, 0.5, heatmap_frame, 0.5, 0)

    # --- Cálculo Métricas ---
    area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
    total_frangos = sum(total_deteccoes.values())
    densidade_total = total_frangos / area_total_m2 if area_total_m2 > 0 else 0

    if len(todos_centros) >= 3:
        pontos = np.array(todos_centros).astype(np.int32)
        hull = cv2.convexHull(pontos)
        mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        area_detectada_px = np.count_nonzero(mask)
        m2_por_px2 = area_total_m2 / (altura_real_px * largura_real_px)
        area_detectada_m2 = area_detectada_px * m2_por_px2
        densidade_detectada = total_frangos / area_detectada_m2 if area_detectada_m2 > 0 else 0
    else:
        area_detectada_m2 = 0
        densidade_detectada = 0

    # --- Texto Final ---
    resultado_txt = ''
    resultado_txt += f"\n--- MÉTRICAS DE DENSIDADE ---\n"
    resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
    resultado_txt += f"Área total da imagem (real): {area_total_m2:.2f} m²\n"
    resultado_txt += f"Área com frangos detectados (convexa): {area_detectada_m2:.2f} m²\n"
    resultado_txt += f"Densidade total da imagem: {densidade_total:.2f} frangos/m²\n"
    resultado_txt += f"Densidade apenas onde há frangos: {densidade_detectada:.2f} frangos/m²"

    # --- Codifica as imagens ---
    _, img_boxes_encoded = cv2.imencode('.jpg', frame_boxes)
    _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_blend)

    return img_boxes_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt





# DBSCAN
# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_aglomeracao):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     altura, largura, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])
#     escala = 300
#     largura_m = max(superior, inferior)
#     altura_m = max(esquerdo, direito)
#     largura_real_px = int(largura_m * escala)
#     altura_real_px = int(altura_m * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(
#         source=img,
#         conf=0.7,
#         iou=0.7,
#         imgsz=1280,
#         save=False,
#         save_txt=False,
#         save_conf=False,
#         max_det=1500,
#         line_width=1
#     )

#     resultado = results[0]
#     total_deteccoes = {}
#     todos_centros = []
#     centros_px = []

#     frame_boxes = img.copy()
#     heatmap_frame = np.zeros_like(img)

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         classes_detectadas = resultado.boxes.cls.tolist()

#         for i, classe_id in enumerate(classes_detectadas):
#             classe_id = int(classe_id)
#             nome_classe = nomes_classes[classe_id] if 0 <= classe_id < len(nomes_classes) else f"Classe {classe_id}"
#             total_deteccoes[nome_classe] = total_deteccoes.get(nome_classe, 0) + 1

#             x1, y1, x2, y2 = boxes[i]
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido.tolist())
#             centros_px.append(centro_corrigido.astype(int).tolist())

#             cv2.circle(frame_boxes, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
#             cv2.rectangle(frame_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

#     # --- Clustering DBSCAN ---
#     if todos_centros:
#         limites = {
#             'verde': 7,
#             'amarelo': 8,
#             'laranja': 9,
#             'vermelho': 10
#         }
#         X = np.array(todos_centros)
#         eps_metros = 1
#         eps = eps_metros * 150  # Convertendo metros para pixels
#         for cor_nome, min_samples in limites.items():
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#             labels = clustering.labels_
#             area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))  # já em m²
#             area_total_px = altura_real_px * largura_real_px
#             m2_por_px2 = area_total_m2 / area_total_px

#             for label in set(labels):
#                 if label == -1:
#                     continue  # ruído

#                 pontos_cluster = X[labels == label]
#                 if len(pontos_cluster) < min_samples:
#                     continue 
#                 hull = cv2.convexHull(pontos_cluster.astype(np.int32))
#                 mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, hull, 255)

#                 area_cluster_px2 = np.count_nonzero(mask)
#                 area_cluster_m2 = area_cluster_px2 * m2_por_px2

#                 densidade_cluster = len(pontos_cluster) / area_cluster_m2 if area_cluster_m2 > 0 else 0

#                 cor = {
#                     'vermelho': (0, 0, 255),
#                     'laranja': (0, 128, 255),
#                     'amarelo': (0, 255, 255),
#                     'verde': (0, 255, 0)
#                 }[cor_nome]

#                 hull_pts = hull.squeeze().astype(np.float32).reshape(1, -1, 2)
#                 hull_original = cv2.perspectiveTransform(hull_pts, H_inv).astype(int)
#                 cv2.fillConvexPoly(heatmap_frame, hull_original[0], cor)

#     # --- Geração imagem heatmap sobreposta ---
#     heatmap_blend = cv2.addWeighted(img, 0.5, heatmap_frame, 0.5, 0)

#     # --- Cálculo Métricas ---
#     area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
#     total_frangos = sum(total_deteccoes.values())
#     densidade_total = total_frangos / area_total_m2 if area_total_m2 > 0 else 0

#     if len(todos_centros) >= limite_aglomeracao:
#         pontos = np.array(todos_centros).astype(np.int32)
#         hull = cv2.convexHull(pontos)
#         mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
#         cv2.fillConvexPoly(mask, hull, 255)
#         area_detectada_px = np.count_nonzero(mask)
#         area_detectada_m2 = area_detectada_px * m2_por_px2
#         densidade_detectada = total_frangos / area_detectada_m2 if area_detectada_m2 > 0 else 0
#     else:
#         area_detectada_m2 = 0
#         densidade_detectada = 0

#     # --- Texto Final ---
#     resultado_txt = ''
#     resultado_txt += f"\n--- MÉTRICAS DE DENSIDADE ---\n"
#     resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
#     resultado_txt += f"Área total da imagem (real): {area_total_m2:.2f} m²\n"
#     resultado_txt += f"Área com frangos detectados (convexa): {area_detectada_m2:.2f} m²\n"
#     resultado_txt += f"Densidade total da imagem: {densidade_total:.2f} frangos/m²\n"
#     resultado_txt += f"Densidade apenas onde há frangos: {densidade_detectada:.2f} frangos/m²"

#     # --- Codifica as imagens ---
#     _, img_boxes_encoded = cv2.imencode('.jpg', frame_boxes)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_blend)

#     return img_boxes_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt



# KDTree
# from scipy.spatial import KDTree

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_vizinhos, raio_aglomeracao_cm=60):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")
    
#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")
    
#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])
    
#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     # YOLO detections
#     try:
#         results = model.track(source=img, conf=0.77, iou=0.7, imgsz=1280, save=False, max_det=1500)
#         resultado = results[0]
#     except NameError:
#         raise RuntimeError("Modelo YOLO não definido.")
    
#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)
    
#     todos_centros = np.array(todos_centros)
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) > 0:
#         tree = KDTree(todos_centros)
#         for centro in todos_centros:
#             raio_px = (raio_aglomeracao_cm / 100) * escala
#             vizinhos = tree.query_ball_point(centro, r=raio_px)
#             qtd_vizinhos = len(vizinhos)
#             cor = None
#             if qtd_vizinhos >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             if cor is not None:
#                 cx, cy = centro.astype(int)
#                 if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                     cv2.circle(heatmap_real, (cx, cy), int(0.23 * escala), cor, -1)

#     # Projetar o heatmap de volta para a imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenhar centros detectados
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     # Texto de saída
#     resultado_txt = f"--- ANÁLISE DE AGLOMERAÇÃO DINÂMICA KDTree ---\n"
#     resultado_txt += f"Total de frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Parâmetros: Limiar = {limite_vizinhos} frangos, Raio = {raio_aglomeracao_cm} cm.\n"

#     # Codificar imagem
#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt




# AUTOMATO CEULAR 
# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     # Ajusta os pontos reais para a escala de pixels desejada
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_vizinhos, raio_aglomeracao_cm = 100):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     # --- Homografia: Ajustado para o modelo retangular (o mais robusto sem coordenadas exatas) ---
#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
    
#     largura_total_real_m = superior 
#     altura_total_real_m = max(esquerdo, direito)

#     pontos_reais = np.float32([
#         [0, 0],                               
#         [superior, 0],            
#         [superior, direito], 
#         [0, esquerdo]              
#     ])
    
#     escala = 300  # pixels por metro
#     largura_real_px = int(largura_total_real_m * escala)
#     altura_real_px = int(altura_total_real_m * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H) 
    
#     try:
#         results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#         resultado = results[0]
#     except NameError:
#         print("YOLO model not defined, using mock results.")

#     todos_centros = []

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for i in range(len(boxes)):
#             x1, y1, x2, y2 = boxes[i]
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)

#     # --- Lógica de Autômatos Celulares ---
#     grid_ca = np.zeros((altura_real_px, largura_real_px), dtype=np.float32)

#     for cx, cy in todos_centros:
#         if 0 <= int(cy) < altura_real_px and 0 <= int(cx) < largura_real_px:
#             grid_ca[int(cy), int(cx)] = 1.0

#     tamanho_kernel = int((raio_aglomeracao_cm / 100) * escala)
#     if tamanho_kernel % 2 == 0:
#         tamanho_kernel += 1
    
#     kernel = np.ones((tamanho_kernel, tamanho_kernel), dtype=np.float32)
    
#     mapa_densidade = cv2.filter2D(grid_ca, -1, kernel)

    
#     # --- Geração do Heatmap Personalizado ---
#     # Criar uma máscara colorida inicialmente preta
#     custom_heatmap_real_world = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)
#     for cx, cy in todos_centros:
#         cv2.circle(custom_heatmap_real_world, (int(cx), int(cy)), 4, (255, 255, 255), -1)

#     threshold_vermelho = limite_vizinhos
#     threshold_laranja = limite_vizinhos * 0.90
#     threshold_amarelo = limite_vizinhos * 0.80
#     threshold_verde = limite_vizinhos * 0.70

#     # Definição das cores (BGR no OpenCV)
#     COLOR_VERMELHO = (0, 0, 255)    # BGR
#     COLOR_LARANJA = (0, 165, 255)   # BGR
#     COLOR_AMARELO = (0, 255, 255)   # BGR
#     COLOR_VERDE = (0, 255, 0)       # BGR

#     # Aplicar as cores com base nos limiares (do mais alto para o mais baixo para sobreposição correta)
    
#     # Máscara para VERMELHO (densidade >= limite_vizinhos)
#     mask_vermelho = (mapa_densidade >= threshold_vermelho)
#     custom_heatmap_real_world[mask_vermelho] = COLOR_VERMELHO

#     # Máscara para LARANJA (densidade >= 90% e < limite_vizinhos)
#     mask_laranja = (mapa_densidade >= threshold_laranja) & (mapa_densidade < threshold_vermelho)
#     custom_heatmap_real_world[mask_laranja] = COLOR_LARANJA

#     # Máscara para AMARELO (densidade >= 80% e < 90%)
#     mask_amarelo = (mapa_densidade >= threshold_amarelo) & (mapa_densidade < threshold_laranja)
#     custom_heatmap_real_world[mask_amarelo] = COLOR_AMARELO

#     # Máscara para VERDE (densidade >= 70% e < 80%)
#     mask_verde = (mapa_densidade >= threshold_verde) & (mapa_densidade < threshold_amarelo)
#     custom_heatmap_real_world[mask_verde] = COLOR_VERDE
    
#     # Transformar o heatmap personalizado de volta para a perspectiva da imagem original
#     custom_heatmap_perspectiva = cv2.warpPerspective(
#         custom_heatmap_real_world,
#         H_inv,
#         (largura_img, altura_img),
#         flags=cv2.INTER_LINEAR  # ou cv2.INTER_CUBIC para suavização mais rica
#     )
#     custom_heatmap_perspectiva = cv2.resize(custom_heatmap_perspectiva, (largura_img, altura_img), interpolation=cv2.INTER_LINEAR)

#     # Misturar o heatmap colorido com a imagem original
#     alpha_heatmap = 0.5 # Transparência para o heatmap
#     frame_resultado_final = cv2.addWeighted(custom_heatmap_perspectiva, alpha_heatmap, img.copy(), 1 - alpha_heatmap, 0)
        
#     # Desenhar os centros das galinhas para referência (na imagem original)
#     if resultado.boxes is not None:
#         for i in range(len(resultado.boxes.xyxy.cpu().numpy())):
#             x1, y1, x2, y2 = resultado.boxes.xyxy.cpu().numpy()[i]
#             cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#             cv2.circle(frame_resultado_final, (cx_img, cy_img), 4, (255, 0, 0), -1) # Pontos azuis para frangos

#     # --- Métricas e Texto Final ---
#     total_frangos = len(todos_centros)
#     resultado_txt = f"--- ANÁLISE DE AGLOMERAÇÃO DINÂMICA ---\n"
#     resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
#     # A contagem de aglomerações (contornos) agora é secundária ao heatmap personalizado.
#     # Você pode remover ou adaptar esta linha se não for mais relevante.
#     resultado_txt += f"Parâmetros: Limiar de {limite_vizinhos} frangos (vermelho), raio de {raio_aglomeracao_cm} cm.\n"
#     mapa_densidade_normalizado = cv2.normalize(mapa_densidade, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     cv2.imwrite("mapa_densidade_debug.jpg", mapa_densidade_normalizado)

#     # Codifica as imagens de resultado
#     _, img_resultado_encoded = cv2.imencode('.jpg', frame_resultado_final) # Imagem com heatmap personalizado
#     # Para o segundo retorno, podemos enviar o mapa de densidade normalizado ou o heatmap personalizado (sem transparência)
#     _, img_densidade_encoded = cv2.imencode('.jpg', custom_heatmap_real_world) # O heatmap personalizado no espaço do mundo real (sem transparência)

#     return img_resultado_encoded.tobytes(), img_densidade_encoded.tobytes(), resultado_txt





# Agglomerative Clustering

# from sklearn.cluster import AgglomerativeClustering

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_vizinhos, distancia_metros=0.35):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")
    
#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)

#     todos_centros = np.array(todos_centros)
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) > 1:
#         clustering = AgglomerativeClustering(
#             n_clusters=None,
#             distance_threshold=distancia_metros * escala,  # distância em metros
#             linkage='single'
#         ).fit(todos_centros)

#         labels = clustering.labels_
#         num_clusters = np.max(labels) + 1

#         for cluster_id in range(num_clusters):
#             indices = np.where(labels == cluster_id)[0]
#             tamanho_cluster = len(indices)

#             cor = None
#             if tamanho_cluster >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif tamanho_cluster >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif tamanho_cluster >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif tamanho_cluster >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             if cor:
#                 for idx in indices:
#                     cx, cy = todos_centros[idx].astype(int)
#                     if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                         cv2.circle(heatmap_real, (cx, cy), int(0.2 * escala), cor, -1)

#     # Projeção do heatmap para imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenho dos centros
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     resultado_txt = f"--- ANÁLISE DE AGRUPAMENTO HIERÁRQUICO ---\n"
#     resultado_txt += f"Total de frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Número de clusters: {num_clusters}\n"
#     resultado_txt += f"Limiar de {limite_vizinhos} frangos por grupo, distância: {distancia_metros:.2f} m"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt



# Alpha Shapes + KDTREE
# import cv2
# import numpy as np
# import alphashape
# from shapely.geometry import Polygon, MultiPoint
# from scipy.spatial import KDTree

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def gerar_mascara_alpha_shape(pontos, largura_px, altura_px, escala=300, alpha=2.5):
#     if len(pontos) < 4:
#         return np.ones((altura_px, largura_px), dtype=np.uint8) * 255

#     coords_metros = [tuple(np.array(p) / escala) for p in pontos]

#     try:
#         alpha_shape = alphashape.alphashape(coords_metros, alpha)
#         if alpha_shape is None or alpha_shape.is_empty:
#             alpha_shape = MultiPoint(coords_metros).convex_hull

#         if isinstance(alpha_shape, Polygon):
#             exterior = np.array([[x * escala, y * escala] for x, y in alpha_shape.exterior.coords]).astype(np.int32)
#             mask = np.zeros((altura_px, largura_px), dtype=np.uint8)
#             cv2.fillPoly(mask, [exterior], 255)
#             return mask
#     except Exception as e:
#         print(f"[ERROR] Falha no alpha shape: {e}")

#     return np.ones((altura_px, largura_px), dtype=np.uint8) * 255

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                         limite_vizinhos, raio_aglomeracao_cm=100, alpha=2.5):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha na decodificação da imagem")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)

#     todos_centros = np.array(todos_centros)
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) >= 2:
#         tree = KDTree(todos_centros)
#         raio_px = (raio_aglomeracao_cm / 100) * escala

#         for centro in todos_centros:
#             qtd_vizinhos = len(tree.query_ball_point(centro, r=raio_px))
#             cx, cy = centro.astype(int)

#             cor = None
#             if qtd_vizinhos >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             if cor and 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                 cv2.circle(heatmap_real, (cx, cy), int(0.22 * escala), cor, -1)

#     # Aplicar máscara alpha shape
#     mask_alpha = gerar_mascara_alpha_shape(todos_centros, largura_real_px, altura_real_px, escala=escala, alpha=alpha)
#     heatmap_real = cv2.bitwise_and(heatmap_real, heatmap_real, mask=mask_alpha)

#     # Projeção do heatmap para imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenho dos frangos
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     # Cálculo da área
#     coords_metros = np.array(todos_centros) / escala
#     try:
#         alpha_poly = alphashape.alphashape(coords_metros.tolist(), alpha)
#         area_m2 = alpha_poly.area if alpha_poly else 0
#     except:
#         area_m2 = 0

#     resultado_txt = f"--- DETECÇÃO COM ALPHA SHAPE ---\n"
#     resultado_txt += f"Frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Área estimada ocupada: {area_m2:.2f} m²\n"
#     resultado_txt += f"Raio de análise: {raio_aglomeracao_cm} cm\n"
#     resultado_txt += f"Limite de vizinhos: {limite_vizinhos}\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt



# Kernel Density Estimation (KDE) gaussian
# import cv2
# import numpy as np
# from sklearn.neighbors import KernelDensity
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, bandwidth_m=1.0):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha na decodificação da imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido / escala)  # agora em metros

#     todos_centros = np.array(todos_centros)
#     if len(todos_centros) < 2:
#         raise ValueError("Número insuficiente de frangos para aplicar KDE.")

#     # Kernel Density Estimation
#     kde = KernelDensity(bandwidth=bandwidth_m, kernel='gaussian')
#     kde.fit(todos_centros)

#     # Gerar grade no espaço real
#     x = np.linspace(0, largura_real_px / escala, largura_real_px)
#     y = np.linspace(0, altura_real_px / escala, altura_real_px)
#     X, Y = np.meshgrid(x, y)
#     coords = np.vstack([X.ravel(), Y.ravel()]).T

#     Z = kde.score_samples(coords)
#     Z = np.exp(Z).reshape((altura_real_px, largura_real_px))

#     # Normalizar para 0-255
#     Z_norm = MinMaxScaler().fit_transform(Z)
#     Z_255 = (Z_norm * 255).astype(np.uint8)
#     heatmap_real = cv2.applyColorMap(Z_255, cv2.COLORMAP_JET)

#     # Projetar de volta para a imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenhar frangos detectados
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 255, 255), -1)

#     # Métricas
#     area_total_m2 = (largura_real_px / escala) * (altura_real_px / escala)
#     densidade_media = len(todos_centros) / area_total_m2 if area_total_m2 > 0 else 0

#     resultado_txt = "--- HEATMAP CONTÍNUO COM KDE ---\n"
#     resultado_txt += f"Frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Área analisada: {area_total_m2:.2f} m²\n"
#     resultado_txt += f"Densidade média estimada: {densidade_media:.2f} frangos/m²\n"
#     resultado_txt += f"Bandwidth (suavização): {bandwidth_m:.2f} m\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt






#Delaunay Triangulation + análise de área local
# import cv2
# import numpy as np
# from scipy.spatial import Delaunay

# def calcular_homografia(pontos_imagem, pontos_reais):
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais)
#     return H

# def area_triangulo(tri):
#     a, b, c = tri
#     return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                     limite):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Erro ao decodificar a imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     H = calcular_homografia(pontos_imagem, pontos_reais)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)

#     todos_centros = np.array(todos_centros)
#     img_heatmap = img.copy()
#     triangulos_aglomerados = 0

#     if len(todos_centros) >= 3:
#         delaunay = Delaunay(todos_centros)
#         for tri_idx in delaunay.simplices:
#             tri_pts_real = todos_centros[tri_idx]
#             area_m2 = area_triangulo(tri_pts_real)

#             densidade = 3 / area_m2 if area_m2 > 0 else 0

#             if densidade >= limite:
#                 cor = (0, 0, 255)  # vermelho
#                 triangulos_aglomerados += 1
#             elif densidade >= limite * 0.9:
#                 cor = (0, 165, 255)  # laranja
#             elif densidade >= limite * 0.8:
#                 cor = (0, 255, 255)  # amarelo
#             elif densidade >= limite * 0.7:
#                 cor = (0, 255, 0)  # verde
#             else:
#                 cor = (90, 90, 90)  # cinza claro (não destaca)

#             # Reprojetar triângulo de volta para a imagem original
#             tri_pts_img = cv2.perspectiveTransform(np.array([tri_pts_real], dtype=np.float32), H_inv)[0]
#             tri_pts_img = tri_pts_img.astype(np.int32)
#             cv2.fillPoly(img_heatmap, [tri_pts_img], cor)

#     # Sobrepor com imagem original
#     img_com_heatmap = cv2.addWeighted(img_heatmap, 0.5, img.copy(), 0.5, 0)

#     # Marcar os frangos
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 255, 255), -1)

#     resultado_txt = "--- DELAUNAY TRIANGULATION (DENSIDADE LOCAL) ---\n"
#     resultado_txt += f"Frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Triângulos com densidade ≥ {limite} frangos/m²: {triangulos_aglomerados}\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', img_heatmap)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt




#  Graph-based Clustering (componentes conexas / grafos)
# import cv2
# import numpy as np
# import networkx as nx
# from scipy.spatial.distance import cdist

# def calcular_homografia(pontos_imagem, pontos_reais):
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                   limite, raio_metros=2.0):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Erro ao decodificar a imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     H = calcular_homografia(pontos_imagem, pontos_reais)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)

#     todos_centros = np.array(todos_centros)  # Converte para metros

#     img_heatmap = img.copy()
#     total_frangos = len(todos_centros)
#     total_grupos = 0

#     if total_frangos >= 2:
#         dist_matrix = cdist(todos_centros, todos_centros)
#         G = nx.Graph()

#         for i in range(total_frangos):
#             G.add_node(i)

#         for i in range(total_frangos):
#             for j in range(i + 1, total_frangos):
#                 if dist_matrix[i, j] <= raio_metros:
#                     G.add_edge(i, j)

#         componentes = list(nx.connected_components(G))
#         total_grupos = len(componentes)

#         # Cores por tamanho
#         cores_por_grupo = {}
#         for grupo in componentes:
#             tamanho = len(grupo)
#             if tamanho >= limite:
#                 cor = (0, 0, 255)      # vermelho
#             elif tamanho >= limite * 0.9:
#                 cor = (0, 165, 255)    # laranja
#             elif tamanho >= limite * 0.8:
#                 cor = (0, 255, 255)    # amarelo
#             elif tamanho >= limite * 0.7:
#                 cor = (0, 255, 0)      # verde
#             else:
#                 cor = (90, 90, 90)     # cinza

#             for idx in grupo:
#                 cores_por_grupo[idx] = cor

#         # Desenho
#         centros_px = (todos_centros * 300).astype(int)
#         for i, (cx, cy) in enumerate(centros_px):
#             if 0 <= cx < largura_img and 0 <= cy < altura_img:
#                 cor = cores_por_grupo.get(i, (90, 90, 90))
#                 cv2.circle(img_heatmap, (cx, cy), 10, cor, -1)
#             else:
#                 print(f"[WARN] Frango {i} com coordenadas fora da imagem: ({cx}, {cy})")

#     # Sobreposição com imagem original
#     img_com_heatmap = cv2.addWeighted(img_heatmap, 0.5, img.copy(), 0.5, 0)
    
#     # Marcar os frangos com ponto central branco
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 6, (255, 255, 255), -1)

#     resultado_txt = "--- AGRUPAMENTO BASEADO EM GRAFOS ---\n"
#     resultado_txt += f"Frangos detectados: {total_frangos}\n"
#     resultado_txt += f"Grupos identificados: {total_grupos}\n"
#     resultado_txt += f"Raio de conexão: {raio_metros:.2f} m\n"
#     resultado_txt += f"Limite para aglomeração: {limite} frangos por grupo\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', img_heatmap)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt




#Watershed Transform (segmentação topológica)
# import cv2
# import numpy as np
# from scipy.ndimage import gaussian_filter
# from scipy import ndimage
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# def calcular_homografia(pontos_imagem, pontos_reais):
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais)
#     return H


# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                       limite, escala=300, raio_kernel_px=1):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Erro ao decodificar a imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     H = calcular_homografia(pontos_imagem, pontos_reais)
#     H_inv = np.linalg.inv(H)

#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     results = model.track(source=img, conf=0.55, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido.astype(int))

#     # Mapa de densidade com picos mais altos
#     densidade = np.zeros((altura_real_px, largura_real_px), dtype=np.float32)
#     for (x, y) in todos_centros:
#         if 0 <= y < altura_real_px and 0 <= x < largura_real_px:
#             cv2.circle(densidade, (x, y), radius=6, color=25, thickness=-1)

#     # Suavização
#     densidade_suavizada = gaussian_filter(densidade, sigma=raio_kernel_px)

#     # Tentativa de picos com parâmetros generosos
#     coords = peak_local_max(densidade_suavizada, min_distance=1, threshold_abs=0.01)

#     # Fallback: sem picos detectados → cria picos nos frangos mais densos
#     if coords.shape[0] == 0:
#         print("[WARN] Nenhum marcador detectado. Criando marcadores manualmente...")
#         flat_indices = np.argpartition(densidade_suavizada.ravel(), -5)[-5:]
#         coords = np.array(np.unravel_index(flat_indices, densidade_suavizada.shape)).T

#     # Criar marcadores manualmente
#     marcadores = np.zeros_like(densidade, dtype=np.int32)
#     for i, (y, x) in enumerate(coords, start=1):
#         marcadores[y, x] = i

#     labels = watershed(-densidade_suavizada, marcadores, mask=densidade > 0)
#     num_segmentos = np.max(labels)

#     # Mapeia frangos por região
#     mapa_regioes = {}
#     for idx, (x, y) in enumerate(todos_centros):
#         if 0 <= y < altura_real_px and 0 <= x < largura_real_px:
#             regiao = labels[y, x]
#             if regiao > 0:
#                 mapa_regioes.setdefault(regiao, []).append((x, y))

#     # Heatmap colorido por densidade
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)
#     for regiao, frangos in mapa_regioes.items():
#         qtd = len(frangos)
#         mask = (labels == regiao)

#         if qtd >= limite:
#             cor = (0, 0, 255)
#         elif qtd >= limite * 0.9:
#             cor = (0, 165, 255)
#         elif qtd >= limite * 0.8:
#             cor = (0, 255, 255)
#         elif qtd >= limite * 0.7:
#             cor = (0, 255, 0)
#         else:
#             cor = (100, 100, 100)

#         heatmap_real[mask] = cor

#     # Projeção de volta à imagem original
#     heatmap_img = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img))
#     resultado_final = cv2.addWeighted(heatmap_img, 0.5, img.copy(), 0.5, 0)

#     # Marca frangos
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(resultado_final, (cx_img, cy_img), 3, (255, 255, 255), -1)

#     texto = "--- WATERSHED SEGMENTAÇÃO TOPOGRÁFICA ---\n"
#     texto += f"Frangos detectados: {len(todos_centros)}\n"
#     texto += f"Regiões segmentadas: {num_segmentos}\n"
#     texto += f"Limite: {limite} frangos por região\n"

#     _, img_final_encoded = cv2.imencode('.jpg', resultado_final)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), texto




#DBScan + Agglomerative Clustering
# import numpy as np
# import cv2
# from sklearn.cluster import DBSCAN, AgglomerativeClustering
# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_aglomeracao, distancia_metros=0.35):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_img = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_px = int(max(superior, inferior) * escala)
#     altura_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_img, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     # Detecção de frangos
#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     boxes = resultado.boxes.xyxy.cpu().numpy() if resultado.boxes is not None else []
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#         centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#         todos_centros.append(centro_corrigido)

#     todos_centros = np.array(todos_centros)
#     total_frangos = len(todos_centros)

#     # --- Heatmaps reais (plano corrigido) ---
#     heatmap_db = np.zeros((altura_px, largura_px, 3), dtype=np.uint8)
#     heatmap_ag = np.zeros((altura_px, largura_px, 3), dtype=np.uint8)

#     # --- DBSCAN (com perspectiva) ---
#     if len(todos_centros) > 0:
#         eps = 1.0 * 150
#         limites = {
#             'verde': 7,
#             'amarelo': 8,
#             'laranja': 9,
#             'vermelho': 10
#         }

#         for cor_nome, min_samples in limites.items():
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(todos_centros)
#             labels = clustering.labels_

#             for label in set(labels):
#                 if label == -1:
#                     continue

#                 pontos_cluster = todos_centros[labels == label]
#                 if len(pontos_cluster) < min_samples:
#                     continue

#                 hull = cv2.convexHull(pontos_cluster.astype(np.int32))
#                 mask = np.zeros((altura_px, largura_px), dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, hull, 255)

#                 cor = {
#                     'vermelho': (0, 0, 255),
#                     'laranja': (0, 128, 255),
#                     'amarelo': (0, 255, 255),
#                     'verde': (0, 255, 0)
#                 }[cor_nome]

#                 cv2.fillConvexPoly(heatmap_db, hull, cor)

#     # --- Agglomerative ---
#     if len(todos_centros) > 1:
#         clustering = AgglomerativeClustering(
#             n_clusters=None,
#             distance_threshold=distancia_metros * escala,
#             linkage='single'
#         ).fit(todos_centros)
#         labels = clustering.labels_
#         num_clusters = np.max(labels) + 1

#         for cluster_id in range(num_clusters):
#             indices = np.where(labels == cluster_id)[0]
#             tamanho_cluster = len(indices)
#             cor = None
#             if tamanho_cluster >= limite_aglomeracao:
#                 cor = (0, 0, 255)
#             elif tamanho_cluster >= limite_aglomeracao * 0.9:
#                 cor = (0, 165, 255)
#             elif tamanho_cluster >= limite_aglomeracao * 0.8:
#                 cor = (0, 255, 255)
#             elif tamanho_cluster >= limite_aglomeracao * 0.7:
#                 cor = (0, 255, 0)

#             if cor:
#                 for idx in indices:
#                     cx, cy = todos_centros[idx].astype(int)
#                     if 0 <= cy < altura_px and 0 <= cx < largura_px:
#                         cv2.circle(heatmap_ag, (cx, cy), int(0.2 * escala), cor, -1)

#     # --- Fusão dos heatmaps e projeção ---
#     heatmap_comb = cv2.addWeighted(heatmap_db, 0.5, heatmap_ag, 0.5, 0)
#     heatmap_proj = cv2.warpPerspective(heatmap_comb, H_inv, (largura_img, altura_img))
#     img_resultado = cv2.addWeighted(img, 0.5, heatmap_proj, 0.5, 0)

#     # --- Adiciona bounding boxes e centros ---
#     for box in boxes:
#         x1, y1, x2, y2 = box.astype(int)
#         cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_resultado, (cx, cy), 4, (255, 0, 0), -1)
#         cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)

#     # --- Texto com métricas ---
#     area_real_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
#     densidade_m2 = total_frangos / area_real_m2 if area_real_m2 > 0 else 0

#     resultado_txt = "--- DETECÇÃO HÍBRIDA ---\n"
#     resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
#     resultado_txt += f"Área total corrigida: {area_real_m2:.2f} m²\n"
#     resultado_txt += f"Densidade total: {densidade_m2:.2f} frangos/m²\n"
#     resultado_txt += f"DBSCAN aplicado com homografia\n"
#     resultado_txt += f"Agglomerative Clustering (distância = {distancia_metros:.2f}m)\n"

#     _, img_encoded = cv2.imencode(".jpg", img_resultado)
#     _, heatmap_encoded = cv2.imencode(".jpg", heatmap_proj)

#     return img_encoded.tobytes(), heatmap_encoded.tobytes(), resultado_txt



# DBSCAN + Pintura do Watershed
# import numpy as np
# import cv2
# from sklearn.cluster import DBSCAN
# from scipy.ndimage import gaussian_filter
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais * escala)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_aglomeracao, escala=300):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Erro ao decodificar imagem")

#     altura_img, largura_img, _ = img.shape
#     pontos_img = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     largura_px = int(max(superior, inferior) * escala)
#     altura_px = int(max(esquerdo, direito) * escala)

#     H, _ = cv2.findHomography(pontos_img, pontos_reais * escala)
#     H_inv = np.linalg.inv(H)

#     # Detecção
#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
#             centro_corrigido = cv2.perspectiveTransform(np.array([centro]), H)[0][0]
#             todos_centros.append(centro_corrigido)

#     todos_centros = np.array(todos_centros).astype(int)
#     total_frangos = len(todos_centros)

#     # --- Mapa de densidade contínuo ---
#     densidade = np.zeros((altura_px, largura_px), dtype=np.float32)
#     for (x, y) in todos_centros:
#         if 0 <= y < altura_px and 0 <= x < largura_px:
#             cv2.circle(densidade, (x, y), radius=20, color=255, thickness=-1)

#     densidade = cv2.GaussianBlur(densidade, (51, 51), 0)
#     densidade_suavizada = gaussian_filter(densidade, sigma=3)

#     # Conversão de densidade de "valores de pixel" para frangos/m²
#     area_real_m2 = cv2.contourArea(pontos_reais)
#     area_px2 = altura_px * largura_px
#     m2_por_px2 = area_real_m2 / area_px2

#     densidade_m2 = densidade_suavizada / m2_por_px2

#     # Normalização com base no limite de aglomeração
#     densidade_norm = np.clip((densidade_m2 / limite_aglomeracao) * 255, 0, 255).astype(np.uint8)

#     # Aplicar colormap degradê contínuo
#     heatmap_colorido = cv2.applyColorMap(densidade_norm, cv2.COLORMAP_JET)

#     # Projeção de volta à imagem original
#     heatmap_img = cv2.warpPerspective(heatmap_colorido, H_inv, (largura_img, altura_img))
#     img_resultado = cv2.addWeighted(img, 0.5, heatmap_img, 0.5, 0)

#     # Desenhar frangos detectados
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box.astype(int)
#         cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_resultado, (cx, cy), 3, (255, 255, 255), -1)
#         cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)

#     # Métricas
#     densidade_média_total = total_frangos / area_real_m2 if area_real_m2 > 0 else 0

#     texto = "--- ANÁLISE DE DENSIDADE (HEATMAP CONTÍNUO) ---\n"
#     texto += f"Frangos detectados: {total_frangos}\n"
#     texto += f"Área estimada: {area_real_m2:.2f} m²\n"
#     texto += f"Densidade média: {densidade_média_total:.2f} frangos/m²\n"
#     texto += f"Limite de aglomeração: {limite_aglomeracao} frangos/m²\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_resultado)
#     _, heatmap_encoded = cv2.imencode('.jpg', heatmap_img)

#     return img_final_encoded.tobytes(), heatmap_encoded.tobytes(), texto


#DBScan 4 pontos
# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_aglomeracao):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     altura, largura, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])
#     escala = 300
#     largura_m = max(superior, inferior)
#     altura_m = max(esquerdo, direito)
#     largura_real_px = int(largura_m * escala)
#     altura_real_px = int(altura_m * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(
#         source=img,
#         conf=0.7,
#         iou=0.7,
#         imgsz=1280,
#         save=False,
#         save_txt=False,
#         save_conf=False,
#         max_det=1500,
#         line_width=1
#     )

#     resultado = results[0]
#     total_deteccoes = {}
#     todos_centros = []
#     centros_px = []

#     frame_boxes = img.copy()
#     heatmap_frame = np.zeros_like(img)

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         classes_detectadas = resultado.boxes.cls.tolist()

#         for i, classe_id in enumerate(classes_detectadas):
#             classe_id = int(classe_id)
#             nome_classe = nomes_classes[classe_id] if 0 <= classe_id < len(nomes_classes) else f"Classe {classe_id}"
#             total_deteccoes[nome_classe] = total_deteccoes.get(nome_classe, 0) + 1

#             x1, y1, x2, y2 = boxes[i]

#             pontos_bbox = np.array([
#                 [x1, y1],
#                 [x2, y1],
#                 [x2, y2],
#                 [x1, y2]
#             ], dtype=np.float32)
#             # with open("pontos_bbox.txt", "a") as f:
#             #     for ponto in pontos_bbox:
#             #         f.write(str(ponto) + "\n")  # `str(ponto)` mantém os colchetes
#             #     f.write("\n")

#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]
#             # with open("pontos_corrigidos.txt", "a") as f:
#             #     for ponto in pontos_corrigidos:
#             #         f.write(str(ponto) + "\n")  # `str(ponto)` mantém os colchetes
#             #     f.write("\n")

#             for ponto_corrigido in pontos_corrigidos:
#                 todos_centros.append(ponto_corrigido.tolist())
#                 centros_px.append(ponto_corrigido.astype(int).tolist())

#             cv2.rectangle(frame_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
#             for px, py in pontos_bbox.astype(int):
#                 cv2.circle(frame_boxes, (px, py), 3, (255, 0, 0), -1)

#     # --- Clustering DBSCAN com lógica percentual ---
#     if todos_centros:
#         limites = {
#             'verde': max(1, int(limite_aglomeracao * 0.70)),
#             'amarelo': max(1, int(limite_aglomeracao * 0.80)),
#             'laranja': max(1, int(limite_aglomeracao * 0.90)),
#             'vermelho': max(1, int(limite_aglomeracao))
#         }

#         X = np.array(todos_centros)
#         eps_metros = 1
#         eps = eps_metros * 80

#         for cor_nome, min_samples in limites.items():
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#             labels = clustering.labels_
#             area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
#             area_total_px = altura_real_px * largura_real_px
#             m2_por_px2 = area_total_m2 / area_total_px

#             for label in set(labels):
#                 if label == -1:
#                     continue

#                 pontos_cluster = X[labels == label]
#                 if len(pontos_cluster) < min_samples:
#                     continue

#                 hull = cv2.convexHull(pontos_cluster.astype(np.int32))
#                 mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, hull, 255)

#                 area_cluster_px2 = np.count_nonzero(mask)
#                 area_cluster_m2 = area_cluster_px2 * m2_por_px2
#                 densidade_cluster = len(pontos_cluster) / area_cluster_m2 if area_cluster_m2 > 0 else 0

#                 cor = {
#                     'vermelho': (0, 0, 255),
#                     'laranja': (0, 128, 255),
#                     'amarelo': (0, 255, 255),
#                     'verde': (0, 255, 0)
#                 }[cor_nome]

#                 hull_pts = hull.squeeze().astype(np.float32).reshape(1, -1, 2)
#                 hull_original = cv2.perspectiveTransform(hull_pts, H_inv).astype(int)
#                 cv2.fillConvexPoly(heatmap_frame, hull_original[0], cor)

#     # --- Geração imagem heatmap sobreposta ---
#     heatmap_blend = cv2.addWeighted(img, 0.5, heatmap_frame, 0.5, 0)

#     # --- Cálculo Métricas ---
#     area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
#     total_frangos = sum(total_deteccoes.values())
#     densidade_total = total_frangos / area_total_m2 if area_total_m2 > 0 else 0

#     if len(todos_centros) >= limite_aglomeracao:
#         pontos = np.array(todos_centros).astype(np.int32)
#         hull = cv2.convexHull(pontos)
#         mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
#         cv2.fillConvexPoly(mask, hull, 255)
#         area_detectada_px = np.count_nonzero(mask)
#         m2_por_px2 = area_total_m2 / (altura_real_px * largura_real_px)
#         area_detectada_m2 = area_detectada_px * m2_por_px2
#         densidade_detectada = total_frangos / area_detectada_m2 if area_detectada_m2 > 0 else 0
#     else:
#         area_detectada_m2 = 0
#         densidade_detectada = 0

#     resultado_txt = ''
#     resultado_txt += f"\n--- MÉTRICAS DE DENSIDADE ---\n"
#     resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
#     resultado_txt += f"Área total da imagem (real): {area_total_m2:.2f} m²\n"
#     resultado_txt += f"Área com frangos detectados (convexa): {area_detectada_m2:.2f} m²\n"
#     resultado_txt += f"Densidade total da imagem: {densidade_total:.2f} frangos/m²\n"
#     resultado_txt += f"Densidade apenas onde há frangos: {densidade_detectada:.2f} frangos/m²"

#     _, img_boxes_encoded = cv2.imencode('.jpg', frame_boxes)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_blend)

#     return img_boxes_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt

#DBScan 8 pontos
# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_aglomeracao):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     altura, largura, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])
#     escala = 300
#     largura_m = max(superior, inferior)
#     altura_m = max(esquerdo, direito)
#     largura_real_px = int(largura_m * escala)
#     altura_real_px = int(altura_m * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(
#         source=img,
#         conf=0.7,
#         iou=0.7,
#         imgsz=1280,
#         save=False,
#         save_txt=False,
#         save_conf=False,
#         max_det=1500,
#         line_width=1
#     )

#     resultado = results[0]
#     total_deteccoes = {}
#     todos_centros = []
#     centros_px = []

#     frame_boxes = img.copy()
#     heatmap_frame = np.zeros_like(img)

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         classes_detectadas = resultado.boxes.cls.tolist()

#         for i, classe_id in enumerate(classes_detectadas):
#             classe_id = int(classe_id)
#             nome_classe = nomes_classes[classe_id] if 0 <= classe_id < len(nomes_classes) else f"Classe {classe_id}"
#             total_deteccoes[nome_classe] = total_deteccoes.get(nome_classe, 0) + 1

#             x1, y1, x2, y2 = boxes[i]

#             pontos_bbox = np.array([
#                 [x1, y1],                                 # canto superior esquerdo
#                 [x2, y1],                                 # canto superior direito
#                 [x2, y2],                                 # canto inferior direito
#                 [x1, y2],                                 # canto inferior esquerdo
#                 [(x1 + x2) / 2, y1],                      # meio superior
#                 [x2, (y1 + y2) / 2],                      # meio direito
#                 [(x1 + x2) / 2, y2],                      # meio inferior
#                 [x1, (y1 + y2) / 2]                       # meio esquerdo
#             ], dtype=np.float32)

#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]
#             with open("pontos_corrigidos.txt", "a") as f:
#                 for ponto in pontos_corrigidos:
#                     f.write(str(ponto) + "\n")  # `str(ponto)` mantém os colchetes
#                 f.write("\n")

#             for ponto_corrigido in pontos_corrigidos:
#                 todos_centros.append(ponto_corrigido.tolist())
#                 centros_px.append(ponto_corrigido.astype(int).tolist())

#             cv2.rectangle(frame_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
#             for px, py in pontos_bbox.astype(int):
#                 cv2.circle(frame_boxes, (px, py), 2, (255, 0, 0), -1)

#     # --- Clustering DBSCAN com lógica percentual ---
#     if todos_centros:
#         limites = {
#             'verde': max(1, int(limite_aglomeracao * 0.70)),
#             'amarelo': max(1, int(limite_aglomeracao * 0.80)),
#             'laranja': max(1, int(limite_aglomeracao * 0.90)),
#             'vermelho': max(1, int(limite_aglomeracao))
#         }

#         X = np.array(todos_centros)
#         eps_metros = 1
#         eps = eps_metros * 70

#         for cor_nome, min_samples in limites.items():
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#             labels = clustering.labels_
#             area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
#             area_total_px = altura_real_px * largura_real_px
#             m2_por_px2 = area_total_m2 / area_total_px

#             for label in set(labels):
#                 if label == -1:
#                     continue

#                 pontos_cluster = X[labels == label]
#                 if len(pontos_cluster) < min_samples:
#                     continue

#                 hull = cv2.convexHull(pontos_cluster.astype(np.int32))
#                 mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, hull, 255)

#                 area_cluster_px2 = np.count_nonzero(mask)
#                 area_cluster_m2 = area_cluster_px2 * m2_por_px2
#                 densidade_cluster = len(pontos_cluster) / area_cluster_m2 if area_cluster_m2 > 0 else 0

#                 cor = {
#                     'vermelho': (0, 0, 255),
#                     'laranja': (0, 128, 255),
#                     'amarelo': (0, 255, 255),
#                     'verde': (0, 255, 0)
#                 }[cor_nome]

#                 hull_pts = hull.squeeze().astype(np.float32).reshape(1, -1, 2)
#                 hull_original = cv2.perspectiveTransform(hull_pts, H_inv).astype(int)
#                 cv2.fillConvexPoly(heatmap_frame, hull_original[0], cor)

#     # --- Geração imagem heatmap sobreposta ---
#     heatmap_blend = cv2.addWeighted(img, 0.5, heatmap_frame, 0.5, 0)

#     # --- Cálculo Métricas ---
#     area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
#     total_frangos = sum(total_deteccoes.values())
#     densidade_total = total_frangos / area_total_m2 if area_total_m2 > 0 else 0

#     if len(todos_centros) >= limite_aglomeracao:
#         pontos = np.array(todos_centros).astype(np.int32)
#         hull = cv2.convexHull(pontos)
#         mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
#         cv2.fillConvexPoly(mask, hull, 255)
#         area_detectada_px = np.count_nonzero(mask)
#         m2_por_px2 = area_total_m2 / (altura_real_px * largura_real_px)
#         area_detectada_m2 = area_detectada_px * m2_por_px2
#         densidade_detectada = total_frangos / area_detectada_m2 if area_detectada_m2 > 0 else 0
#     else:
#         area_detectada_m2 = 0
#         densidade_detectada = 0

#     resultado_txt = ''
#     resultado_txt += f"\n--- MÉTRICAS DE DENSIDADE ---\n"
#     resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
#     resultado_txt += f"Área total da imagem (real): {area_total_m2:.2f} m²\n"
#     resultado_txt += f"Área com frangos detectados (convexa): {area_detectada_m2:.2f} m²\n"
#     resultado_txt += f"Densidade total da imagem: {densidade_total:.2f} frangos/m²\n"
#     resultado_txt += f"Densidade apenas onde há frangos: {densidade_detectada:.2f} frangos/m²"

#     _, img_boxes_encoded = cv2.imencode('.jpg', frame_boxes)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_blend)

#     return img_boxes_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt

# KDTree 8 pontos
# from scipy.spatial import KDTree

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_vizinhos, raio_aglomeracao_cm=60):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")
    
#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")
    
#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])
    
#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     # YOLO detections
#     try:
#         results = model.track(source=img, conf=0.77, iou=0.7, imgsz=1280, save=False, max_det=1500)
#         resultado = results[0]
#     except NameError:
#         raise RuntimeError("Modelo YOLO não definido.")
    
#     todos_centros = []

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box

#             # Define os 8 pontos do bounding box
#             pontos_bbox = np.array([
#                 [x1, y1],                                 # canto superior esquerdo
#                 [x2, y1],                                 # canto superior direito
#                 [x2, y2],                                 # canto inferior direito
#                 [x1, y2],                                 # canto inferior esquerdo
#                 [(x1 + x2) / 2, y1],                      # meio superior
#                 [x2, (y1 + y2) / 2],                      # meio direito
#                 [(x1 + x2) / 2, y2],                      # meio inferior
#                 [x1, (y1 + y2) / 2]                       # meio esquerdo
#             ], dtype=np.float32)

#             # Aplica a homografia nos 8 pontos
#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]

#             # Adiciona ao array final
#             for ponto in pontos_corrigidos:
#                 todos_centros.append(ponto)

#     # Converte para array numpy final
#     todos_centros = np.array(todos_centros)

#     # Inicializa o heatmap vazio
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) > 0:
#         tree = KDTree(todos_centros)
#         for centro in todos_centros:
#             raio_px = (raio_aglomeracao_cm / 100) * escala
#             vizinhos = tree.query_ball_point(centro, r=raio_px)
#             qtd_vizinhos = len(vizinhos)
#             cor = None
#             if qtd_vizinhos >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             if cor is not None:
#                 cx, cy = centro.astype(int)
#                 if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                     cv2.circle(heatmap_real, (cx, cy), int(0.15 * escala), cor, -1)

#     # Projetar o heatmap de volta para a imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenhar centros detectados
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     # Texto de saída
#     resultado_txt = f"--- ANÁLISE DE AGLOMERAÇÃO DINÂMICA KDTree ---\n"
#     resultado_txt += f"Total de frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Parâmetros: Limiar = {limite_vizinhos} frangos, Raio = {raio_aglomeracao_cm} cm.\n"

#     # Codificar imagem
#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt





# AUTOMATO CEULAR 8 PONTOS
# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     # Ajusta os pontos reais para a escala de pixels desejada
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_vizinhos, raio_aglomeracao_cm = 90):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     # --- Homografia: Ajustado para o modelo retangular (o mais robusto sem coordenadas exatas) ---
#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
    
#     largura_total_real_m = superior 
#     altura_total_real_m = max(esquerdo, direito)

#     pontos_reais = np.float32([
#         [0, 0],                               
#         [superior, 0],            
#         [superior, direito], 
#         [0, esquerdo]              
#     ])
    
#     escala = 300  # pixels por metro
#     largura_real_px = int(largura_total_real_m * escala)
#     altura_real_px = int(altura_total_real_m * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H) 
    
#     try:
#         results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#         resultado = results[0]
#     except NameError:
#         print("YOLO model not defined, using mock results.")

#     todos_centros = []

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box

#             # Define os 8 pontos do bounding box
#             pontos_bbox = np.array([
#                 [x1, y1],                                 # canto superior esquerdo
#                 [x2, y1],                                 # canto superior direito
#                 [x2, y2],                                 # canto inferior direito
#                 [x1, y2],                                 # canto inferior esquerdo
#                 [(x1 + x2) / 2, y1],                      # meio superior
#                 [x2, (y1 + y2) / 2],                      # meio direito
#                 [(x1 + x2) / 2, y2],                      # meio inferior
#                 [x1, (y1 + y2) / 2]                       # meio esquerdo
#             ], dtype=np.float32)

#             # Aplica a homografia nos 8 pontos
#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]
#             # Adiciona ao array final
#             for ponto in pontos_corrigidos:
#                 todos_centros.append(ponto)

#     # Converte para array numpy final
#     todos_centros = np.array(todos_centros)

#     # Inicializa o heatmap vazio

#     # --- Lógica de Autômatos Celulares ---
#     grid_ca = np.zeros((altura_real_px, largura_real_px), dtype=np.float32)

#     for cx, cy in todos_centros:
#         if 0 <= int(cy) < altura_real_px and 0 <= int(cx) < largura_real_px:
#             grid_ca[int(cy), int(cx)] = .8

#     tamanho_kernel = int((raio_aglomeracao_cm / 100) * 300)
#     if tamanho_kernel % 2 == 0:
#         tamanho_kernel += 1
    
#     kernel = np.ones((tamanho_kernel, tamanho_kernel), dtype=np.float32)
    
#     mapa_densidade = cv2.filter2D(grid_ca, -1, kernel)

    
#     # --- Geração do Heatmap Personalizado ---
#     # Criar uma máscara colorida inicialmente preta
#     custom_heatmap_real_world = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)
#     for cx, cy in todos_centros:
#         cv2.circle(custom_heatmap_real_world, (int(cx), int(cy)), 4, (255, 255, 255), -1)

#     threshold_vermelho = limite_vizinhos
#     threshold_laranja = limite_vizinhos * 0.90
#     threshold_amarelo = limite_vizinhos * 0.85
#     threshold_verde = limite_vizinhos * 0.80

#     # Definição das cores (BGR no OpenCV)
#     COLOR_VERMELHO = (0, 0, 255)    # BGR
#     COLOR_LARANJA = (0, 165, 255)   # BGR
#     COLOR_AMARELO = (0, 255, 255)   # BGR
#     COLOR_VERDE = (0, 255, 0)       # BGR

#     # Aplicar as cores com base nos limiares (do mais alto para o mais baixo para sobreposição correta)
    
#     # Máscara para VERMELHO (densidade >= limite_vizinhos)
#     mask_vermelho = (mapa_densidade >= threshold_vermelho)
#     custom_heatmap_real_world[mask_vermelho] = COLOR_VERMELHO

#     # Máscara para LARANJA (densidade >= 90% e < limite_vizinhos)
#     mask_laranja = (mapa_densidade >= threshold_laranja) & (mapa_densidade < threshold_vermelho)
#     custom_heatmap_real_world[mask_laranja] = COLOR_LARANJA

#     # Máscara para AMARELO (densidade >= 80% e < 90%)
#     mask_amarelo = (mapa_densidade >= threshold_amarelo) & (mapa_densidade < threshold_laranja)
#     custom_heatmap_real_world[mask_amarelo] = COLOR_AMARELO

#     # Máscara para VERDE (densidade >= 70% e < 80%)
#     mask_verde = (mapa_densidade >= threshold_verde) & (mapa_densidade < threshold_amarelo)
#     custom_heatmap_real_world[mask_verde] = COLOR_VERDE
    
#     # Transformar o heatmap personalizado de volta para a perspectiva da imagem original
#     custom_heatmap_perspectiva = cv2.warpPerspective(
#         custom_heatmap_real_world,
#         H_inv,
#         (largura_img, altura_img),
#         flags=cv2.INTER_LINEAR  # ou cv2.INTER_CUBIC para suavização mais rica
#     )
#     custom_heatmap_perspectiva = cv2.resize(custom_heatmap_perspectiva, (largura_img, altura_img), interpolation=cv2.INTER_LINEAR)

#     # Misturar o heatmap colorido com a imagem original
#     alpha_heatmap = 0.5 # Transparência para o heatmap
#     frame_resultado_final = cv2.addWeighted(custom_heatmap_perspectiva, alpha_heatmap, img.copy(), 1 - alpha_heatmap, 0)
        
#     # Desenhar os centros das galinhas para referência (na imagem original)
#     if resultado.boxes is not None:
#         for i in range(len(resultado.boxes.xyxy.cpu().numpy())):
#             x1, y1, x2, y2 = resultado.boxes.xyxy.cpu().numpy()[i]
#             cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#             cv2.circle(frame_resultado_final, (cx_img, cy_img), 4, (255, 0, 0), -1) # Pontos azuis para frangos

#     # --- Métricas e Texto Final ---
#     total_frangos = len(todos_centros)
#     resultado_txt = f"--- ANÁLISE DE AGLOMERAÇÃO DINÂMICA ---\n"
#     resultado_txt += f"Total de frangos detectados: {total_frangos}\n"
#     # A contagem de aglomerações (contornos) agora é secundária ao heatmap personalizado.
#     # Você pode remover ou adaptar esta linha se não for mais relevante.
#     resultado_txt += f"Parâmetros: Limiar de {limite_vizinhos} frangos (vermelho), raio de {raio_aglomeracao_cm} cm.\n"
#     mapa_densidade_normalizado = cv2.normalize(mapa_densidade, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     cv2.imwrite("mapa_densidade_debug.jpg", mapa_densidade_normalizado)

#     # Codifica as imagens de resultado
#     _, img_resultado_encoded = cv2.imencode('.jpg', frame_resultado_final) # Imagem com heatmap personalizado
#     # Para o segundo retorno, podemos enviar o mapa de densidade normalizado ou o heatmap personalizado (sem transparência)
#     _, img_densidade_encoded = cv2.imencode('.jpg', custom_heatmap_real_world) # O heatmap personalizado no espaço do mundo real (sem transparência)

#     return img_resultado_encoded.tobytes(), img_densidade_encoded.tobytes(), resultado_txt





# Agglomerative Clustering 8 PONTOS

# from sklearn.cluster import AgglomerativeClustering

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, limite_vizinhos, distancia_metros=0.35):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")
    
#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha ao decodificar imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box

#             # Define os 8 pontos do bounding box
#             pontos_bbox = np.array([
#                 [x1, y1],                                 # canto superior esquerdo
#                 [x2, y1],                                 # canto superior direito
#                 [x2, y2],                                 # canto inferior direito
#                 [x1, y2],                                 # canto inferior esquerdo
#                 [(x1 + x2) / 2, y1],                      # meio superior
#                 [x2, (y1 + y2) / 2],                      # meio direito
#                 [(x1 + x2) / 2, y2],                      # meio inferior
#                 [x1, (y1 + y2) / 2]                       # meio esquerdo
#             ], dtype=np.float32)

#             # Aplica a homografia nos 8 pontos
#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]

#             # Adiciona ao array final
#             for ponto in pontos_corrigidos:
#                 todos_centros.append(ponto)

#     # Converte para array numpy final
#     todos_centros = np.array(todos_centros)

#     # Inicializa o heatmap vazio
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) > 1:
#         clustering = AgglomerativeClustering(
#             n_clusters=None,
#             distance_threshold=distancia_metros * escala,  # distância em metros
#             linkage='single'
#         ).fit(todos_centros)

#         labels = clustering.labels_
#         num_clusters = np.max(labels) + 1

#         for cluster_id in range(num_clusters):
#             indices = np.where(labels == cluster_id)[0]
#             tamanho_cluster = len(indices)

#             cor = None
#             if tamanho_cluster >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif tamanho_cluster >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif tamanho_cluster >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif tamanho_cluster >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             if cor:
#                 for idx in indices:
#                     cx, cy = todos_centros[idx].astype(int)
#                     if 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                         cv2.circle(heatmap_real, (cx, cy), int(0.2 * escala), cor, -1)

#     # Projeção do heatmap para imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenho dos centros
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     resultado_txt = f"--- ANÁLISE DE AGRUPAMENTO HIERÁRQUICO ---\n"
#     resultado_txt += f"Total de frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Número de clusters: {num_clusters}\n"
#     resultado_txt += f"Limiar de {limite_vizinhos} frangos por grupo, distância: {distancia_metros:.2f} m"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt





# Alpha Shapes + KDTREE 8 PONTOS
# import cv2
# import numpy as np
# import alphashape
# from shapely.geometry import Polygon, MultiPoint
# from scipy.spatial import KDTree
# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def gerar_mascara_alpha_shape(pontos, largura_px, altura_px, escala, alpha):
#     import alphashape
#     if len(pontos) < 4:
#         return np.ones((altura_px, largura_px), dtype=np.uint8) * 255

#     coords_metros = [tuple(np.array(p) / escala) for p in pontos]

#     try:
#         alpha_shape = alphashape.alphashape(coords_metros, alpha)
#         if alpha_shape is None or alpha_shape.is_empty:
#             alpha_shape = MultiPoint(coords_metros).convex_hull

#         if isinstance(alpha_shape, Polygon):
#             exterior = np.array([[x * escala, y * escala] for x, y in alpha_shape.exterior.coords]).astype(np.int32)
#             mask = np.zeros((altura_px, largura_px), dtype=np.uint8)
#             cv2.fillPoly(mask, [exterior], 255)
#             return mask
#     except Exception as e:
#         print(f"[ERROR] Falha no alpha shape: {e}")

#     return np.ones((altura_px, largura_px), dtype=np.uint8) * 255

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                         limite_vizinhos, raio_aglomeracao_cm=100, alpha=0.5):
#     import alphashape
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha na decodificação da imagem")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box

#             # Define os 8 pontos do bounding box
#             pontos_bbox = np.array([
#                 [x1, y1],                                 # canto superior esquerdo
#                 [x2, y1],                                 # canto superior direito
#                 [x2, y2],                                 # canto inferior direito
#                 [x1, y2],                                 # canto inferior esquerdo
#                 [(x1 + x2) / 2, y1],                      # meio superior
#                 [x2, (y1 + y2) / 2],                      # meio direito
#                 [(x1 + x2) / 2, y2],                      # meio inferior
#                 [x1, (y1 + y2) / 2]                       # meio esquerdo
#             ], dtype=np.float32)

#             # Aplica a homografia nos 8 pontos
#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]

#             # Adiciona ao array final
#             for ponto in pontos_corrigidos:
#                 todos_centros.append(ponto)

#     # Converte para array numpy final
#     todos_centros = np.array(todos_centros)

#     # Inicializa o heatmap vazio
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) >= 2:
#         tree = KDTree(todos_centros)
#         raio_px = (raio_aglomeracao_cm / 100) * 100

#         for centro in todos_centros:
#             qtd_vizinhos = len(tree.query_ball_point(centro, r=raio_px))
#             cx, cy = centro.astype(int)

#             cor = None
#             if qtd_vizinhos >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif qtd_vizinhos >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             if cor and 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                 cv2.circle(heatmap_real, (cx, cy), int(0.12 * escala), cor, -1)

#     # Aplicar máscara alpha shape
#     mask_alpha = gerar_mascara_alpha_shape(todos_centros, largura_real_px, altura_real_px, escala, alpha)
#     heatmap_real = cv2.bitwise_and(heatmap_real, heatmap_real, mask=mask_alpha)

#     # Projeção do heatmap para imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenho dos frangos
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     # Cálculo da área
#     coords_metros = np.array(todos_centros) / escala
#     try:
#         alpha_poly = alphashape.alphashape(coords_metros.tolist(), alpha)
#         area_m2 = alpha_poly.area if alpha_poly else 0
#     except:
#         area_m2 = 0

#     resultado_txt = f"--- DETECÇÃO COM ALPHA SHAPE ---\n"
#     resultado_txt += f"Frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Área estimada ocupada: {area_m2:.2f} m²\n"
#     resultado_txt += f"Raio de análise: {raio_aglomeracao_cm} cm\n"
#     resultado_txt += f"Limite de vizinhos: {limite_vizinhos}\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt



# FINAL
# import cv2
# import numpy as np
# import alphashape
# from shapely.geometry import Polygon, MultiPoint
# from scipy.spatial import KDTree

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def gerar_mascara_alpha_shape(pontos, largura_px, altura_px, escala, alpha):
#     if len(pontos) < 4:
#         return np.ones((altura_px, largura_px), dtype=np.uint8) * 255

#     coords_metros = [tuple(np.array(p) / escala) for p in pontos]

#     try:
#         alpha_shape = alphashape.alphashape(coords_metros, alpha)
#         if alpha_shape is None or alpha_shape.is_empty:
#             alpha_shape = MultiPoint(coords_metros).convex_hull

#         if isinstance(alpha_shape, Polygon):
#             exterior = np.array([[x * escala, y * escala] for x, y in alpha_shape.exterior.coords]).astype(np.int32)
#             mask = np.zeros((altura_px, largura_px), dtype=np.uint8)
#             cv2.fillPoly(mask, [exterior], 255)
#             return mask
#     except Exception as e:
#         print(f"[ERROR] Falha no alpha shape: {e}")

#     return np.ones((altura_px, largura_px), dtype=np.uint8) * 255

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                           limite_vizinhos, raio_aglomeracao_cm=75, alpha=3.0):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha na decodificação da imagem")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []

#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             pontos_bbox = np.array([
#                 [x1, y1], [x2, y1], [x2, y2], [x1, y2],
#                 [(x1 + x2) / 2, y1], [x2, (y1 + y2) / 2],
#                 [(x1 + x2) / 2, y2], [x1, (y1 + y2) / 2]
#             ], dtype=np.float32)

#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]
#             for ponto in pontos_corrigidos:
#                 todos_centros.append(ponto)

#     todos_centros = np.array(todos_centros)
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)

#     if len(todos_centros) >= 2:
#         tree = KDTree(todos_centros)
#         raio_px = (raio_aglomeracao_cm / 100) * 85

#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             pontos_bbox = np.array([
#                 [x1, y1], [x2, y1], [x2, y2], [x1, y2],
#                 [(x1 + x2) / 2, y1], [x2, (y1 + y2) / 2],
#                 [(x1 + x2) / 2, y2], [x1, (y1 + y2) / 2]
#             ], dtype=np.float32)

#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]
#             densidades = [len(tree.query_ball_point(p, r=raio_px)) for p in pontos_corrigidos]
#             max_vizinhos = max(densidades)

#             cor = None
#             if max_vizinhos >= limite_vizinhos:
#                 cor = (0, 0, 255)
#             elif max_vizinhos >= limite_vizinhos * 0.9:
#                 cor = (0, 165, 255)
#             elif max_vizinhos >= limite_vizinhos * 0.8:
#                 cor = (0, 255, 255)
#             elif max_vizinhos >= limite_vizinhos * 0.7:
#                 cor = (0, 255, 0)

#             centro = np.mean(pontos_corrigidos, axis=0).astype(int)
#             cx, cy = centro
#             if cor and 0 <= cy < altura_real_px and 0 <= cx < largura_real_px:
#                 cv2.circle(heatmap_real, (cx, cy), int(0.14 * escala), cor, -1)

#     mask_alpha = gerar_mascara_alpha_shape(todos_centros, largura_real_px, altura_real_px, escala, alpha)
#     heatmap_real = cv2.bitwise_and(heatmap_real, heatmap_real, mask=mask_alpha)

#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 0, 0), -1)

#     coords_metros = np.array(todos_centros) / escala
#     try:
#         alpha_poly = alphashape.alphashape(coords_metros.tolist(), alpha)
#         area_m2 = alpha_poly.area if alpha_poly else 0
#     except:
#         area_m2 = 0
        
#     total_frangos = len(todos_centros)/8
#     resultado_txt = f"--- DETECÇÃO COM ALPHA SHAPE ---\n"
#     resultado_txt += f"Frangos detectados: {total_frangos}\n"
#     resultado_txt += f"Área estimada ocupada: {area_m2:.2f} m²\n"
#     resultado_txt += f"Raio de análise: {raio_aglomeracao_cm} cm\n"
#     resultado_txt += f"Limite de vizinhos: {limite_vizinhos}\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

    # return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt













# Kernel Density Estimation (KDE) gaussian 8 Pontos
# import cv2
# import numpy as np
# from sklearn.neighbors import KernelDensity
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# def calcular_homografia(pontos_imagem, pontos_reais, escala):
#     pontos_reais_escalados = pontos_reais * escala
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais_escalados)
#     return H

# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior, bandwidth_m=1.0):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo vazio ou não pôde ser lido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Falha na decodificação da imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     escala = 300
#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     H = calcular_homografia(pontos_imagem, pontos_reais, escala)
#     H_inv = np.linalg.inv(H)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box

#             pontos_bbox = np.array([
#                 [x1, y1],                                 # canto superior esquerdo
#                 [x2, y1],                                 # canto superior direito
#                 [x2, y2],                                 # canto inferior direito
#                 [x1, y2],                                 # canto inferior esquerdo
#                 [(x1 + x2) / 2, y1],                      # meio superior
#                 [x2, (y1 + y2) / 2],                      # meio direito
#                 [(x1 + x2) / 2, y2],                      # meio inferior
#                 [x1, (y1 + y2) / 2]                       # meio esquerdo
#             ], dtype=np.float32)

#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]

#             for p_corrigido in pontos_corrigidos:
#                 todos_centros.append(p_corrigido / escala)  # agora em metros


#     todos_centros = np.array(todos_centros)
#     if len(todos_centros) < 2:
#         raise ValueError("Número insuficiente de frangos para aplicar KDE.")

#     # Kernel Density Estimation
#     kde = KernelDensity(bandwidth=bandwidth_m, kernel='gaussian')
#     kde.fit(todos_centros)

#     # Gerar grade no espaço real
#     x = np.linspace(0, largura_real_px / escala, largura_real_px)
#     y = np.linspace(0, altura_real_px / escala, altura_real_px)
#     X, Y = np.meshgrid(x, y)
#     coords = np.vstack([X.ravel(), Y.ravel()]).T

#     Z = kde.score_samples(coords)
#     Z = np.exp(Z).reshape((altura_real_px, largura_real_px))

#     # Normalizar para 0-255
#     Z_norm = MinMaxScaler().fit_transform(Z)
#     Z_255 = (Z_norm * 255).astype(np.uint8)
#     heatmap_real = cv2.applyColorMap(Z_255, cv2.COLORMAP_JET)

#     # Projetar de volta para a imagem original
#     heatmap_proj = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img), flags=cv2.INTER_LINEAR)
#     img_com_heatmap = cv2.addWeighted(heatmap_proj, 0.5, img.copy(), 0.5, 0)

#     # Desenhar frangos detectados
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(img_com_heatmap, (cx_img, cy_img), 4, (255, 255, 255), -1)

#     # Métricas
#     area_total_m2 = (largura_real_px / escala) * (altura_real_px / escala)
#     densidade_media = len(todos_centros) / area_total_m2 if area_total_m2 > 0 else 0

#     resultado_txt = "--- HEATMAP CONTÍNUO COM KDE ---\n"
#     resultado_txt += f"Frangos detectados: {len(todos_centros)}\n"
#     resultado_txt += f"Área analisada: {area_total_m2:.2f} m²\n"
#     resultado_txt += f"Densidade média estimada: {densidade_media:.2f} frangos/m²\n"
#     resultado_txt += f"Bandwidth (suavização): {bandwidth_m:.2f} m\n"

#     _, img_final_encoded = cv2.imencode('.jpg', img_com_heatmap)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), resultado_txt



#Watershed Transform (segmentação topológica)
# import cv2
# import numpy as np
# from scipy.ndimage import gaussian_filter
# from scipy import ndimage
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# def calcular_homografia(pontos_imagem, pontos_reais):
#     H, _ = cv2.findHomography(pontos_imagem, pontos_reais)
#     return H


# def detectar_image_memoria_metro_quadrado(file_storage, superior, esquerdo, direito, inferior,
#                                       limite, escala=300, raio_kernel_px=1):
#     file_storage.stream.seek(0)
#     file_bytes = file_storage.read()
#     if not file_bytes:
#         raise ValueError("Arquivo inválido.")

#     np_arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Erro ao decodificar a imagem.")

#     altura_img, largura_img, _ = img.shape
#     pontos_imagem = np.float32([[0, 0], [largura_img, 0], [largura_img, altura_img], [0, altura_img]])
#     pontos_reais = np.float32([[0, 0], [superior, 0], [inferior, direito], [0, esquerdo]])

#     H = calcular_homografia(pontos_imagem, pontos_reais)
#     H_inv = np.linalg.inv(H)

#     largura_real_px = int(max(superior, inferior) * escala)
#     altura_real_px = int(max(esquerdo, direito) * escala)

#     results = model.track(source=img, conf=0.7, iou=0.7, imgsz=1280, save=False, max_det=1500)
#     resultado = results[0]

#     todos_centros = []
#     if resultado.boxes is not None:
#         boxes = resultado.boxes.xyxy.cpu().numpy()
#         for box in boxes:
#             x1, y1, x2, y2 = box

#             # 8 pontos do bounding box
#             pontos_bbox = np.array([
#                 [x1, y1],
#                 [x2, y1],
#                 [x2, y2],
#                 [x1, y2],
#                 [(x1 + x2) / 2, y1],
#                 [x2, (y1 + y2) / 2],
#                 [(x1 + x2) / 2, y2],
#                 [x1, (y1 + y2) / 2]
#             ], dtype=np.float32)

#             pontos_corrigidos = cv2.perspectiveTransform(np.array([pontos_bbox]), H)[0]

#             for ponto in pontos_corrigidos:
#                 todos_centros.append(ponto.astype(int))

#     # Mapa de densidade com picos mais altos
#     densidade = np.zeros((altura_real_px, largura_real_px), dtype=np.float32)
#     for (x, y) in todos_centros:
#         if 0 <= y < altura_real_px and 0 <= x < largura_real_px:
#             cv2.circle(densidade, (x, y), radius=6, color=25, thickness=-1)

#     # Suavização
#     densidade_suavizada = gaussian_filter(densidade, sigma=raio_kernel_px)

#     # Tentativa de picos com parâmetros generosos
#     coords = peak_local_max(densidade_suavizada, min_distance=1, threshold_abs=0.01)

#     # Fallback: sem picos detectados → cria picos nos frangos mais densos
#     if coords.shape[0] == 0:
#         print("[WARN] Nenhum marcador detectado. Criando marcadores manualmente...")
#         flat_indices = np.argpartition(densidade_suavizada.ravel(), -5)[-5:]
#         coords = np.array(np.unravel_index(flat_indices, densidade_suavizada.shape)).T

#     # Criar marcadores manualmente
#     marcadores = np.zeros_like(densidade, dtype=np.int32)
#     for i, (y, x) in enumerate(coords, start=1):
#         marcadores[y, x] = i

#     labels = watershed(-densidade_suavizada, marcadores, mask=densidade > 0)
#     num_segmentos = np.max(labels)

#     # Mapeia frangos por região
#     mapa_regioes = {}
#     for idx, (x, y) in enumerate(todos_centros):
#         if 0 <= y < altura_real_px and 0 <= x < largura_real_px:
#             regiao = labels[y, x]
#             if regiao > 0:
#                 mapa_regioes.setdefault(regiao, []).append((x, y))

#     # Heatmap colorido por densidade
#     heatmap_real = np.zeros((altura_real_px, largura_real_px, 3), dtype=np.uint8)
#     for regiao, frangos in mapa_regioes.items():
#         qtd = len(frangos)
#         mask = (labels == regiao)

#         if qtd >= limite:
#             cor = (0, 0, 255)
#         elif qtd >= limite * 0.9:
#             cor = (0, 165, 255)
#         elif qtd >= limite * 0.8:
#             cor = (0, 255, 255)
#         elif qtd >= limite * 0.7:
#             cor = (0, 255, 0)
#         else:
#             cor = (100, 100, 100)

#         heatmap_real[mask] = cor

#     # Projeção de volta à imagem original
#     heatmap_img = cv2.warpPerspective(heatmap_real, H_inv, (largura_img, altura_img))
#     resultado_final = cv2.addWeighted(heatmap_img, 0.5, img.copy(), 0.5, 0)

#     # Marca frangos
#     for box in resultado.boxes.xyxy.cpu().numpy():
#         x1, y1, x2, y2 = box
#         cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         cv2.circle(resultado_final, (cx_img, cy_img), 3, (255, 255, 255), -1)

#     texto = "--- WATERSHED SEGMENTAÇÃO TOPOGRÁFICA ---\n"
#     texto += f"Frangos detectados: {len(todos_centros)}\n"
#     texto += f"Regiões segmentadas: {num_segmentos}\n"
#     texto += f"Limite: {limite} frangos por região\n"

#     _, img_final_encoded = cv2.imencode('.jpg', resultado_final)
#     _, img_heatmap_encoded = cv2.imencode('.jpg', heatmap_real)

#     return img_final_encoded.tobytes(), img_heatmap_encoded.tobytes(), texto