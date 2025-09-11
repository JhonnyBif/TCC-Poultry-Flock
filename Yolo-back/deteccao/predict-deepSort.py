# import cv2
# import torch
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from ultralytics import YOLO
# from ultralytics import YOLO, settings
# settings
# settings.update({'weights_dir': './weights'})
# settings.update({'datasets_dir': './datasets'})
# settings.update({'runs_dir': './runs'})

# # Carregar modelo YOLOv11
# model = YOLO('runs/detect/train/weights/best.pt')

# # Inicializar DeepSORT
# tracker = DeepSort(max_age=30)

# # Abrir vídeo
# cap = cv2.VideoCapture("video/frangos_inferior-VEED.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Rodar YOLOv11 no frame
#     results = model(frame)

#     detections = []
#     for result in results:
#         for box in result.boxes.data:
#             x1, y1, x2, y2, score, class_id = box.cpu().numpy()
#             if score > 0.3:  # Ajuste de confiança
#                 detections.append(([x1, y1, x2, y2], score, int(class_id)))

#     # Atualizar rastreamento com DeepSORT
#     tracks = tracker.update_tracks(detections, frame=frame)

#     # Desenhar caixas e IDs
#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         ltrb = track.to_ltrb()
#         x1, y1, x2, y2 = map(int, ltrb)

#         # Desenhar retângulo e ID
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow("Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Carregar modelo YOLO
model = YOLO('runs/detect/train/weights/best.pt')

# Inicializar DeepSORT
tracker = DeepSort(max_age=50,  # Reduz IDs fantasmas
                   n_init=10,  # Garante que apenas rastreamentos confirmados apareçam
                   max_cosine_distance=0.2,# Evita trocas erradas de ID
                   nms_max_overlap = 0.5,
                   embedder="mobilenet")

# Abrir vídeo
video_path = "video/frangos_inferior-VEED.mp4"
cap = cv2.VideoCapture(video_path)

# Configurar saída de vídeo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "resultados/frangos_trackeados_corrigido.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar YOLO no frame
    results = model.predict(frame, conf=0.7, iou=0.5, imgsz=640)

    detections = []
    bounding_boxes = []  # Armazena bounding boxes exatas para desenho

    for result in results:
        if result.boxes is not None:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, class_id = box.cpu().numpy()

                if score > 0.3:  # Apenas detecções confiáveis
                    detections.append(([x1, y1, x2, y2], score, int(class_id)))
                    bounding_boxes.append((x1, y1, x2, y2))  # Guardar caixas reais

    # Atualizar rastreamento
    tracks = tracker.update_tracks(detections, frame=frame)

    # Desenhar caixas e IDs corretamente
    for track, bbox in zip(tracks, bounding_boxes):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, bbox)  # Pegamos exatamente as caixas do YOLO

        # Desenhar bounding box correta
        cv2.rectangle(frame, (x1, y1), (x2, y2), (190, 0, 190), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 0, 190), 2)

    # Salvar frame corrigido
    out.write(frame)
    cv2.imshow("Tracking Ajustado", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo salvo em: {output_path}")


# import cv2
# import torch
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # Carregar modelo YOLO
# model = YOLO('runs/detect/train/weights/best.pt')

# # Inicializar DeepSORT
# tracker = DeepSort(max_age=50,  # Reduz IDs fantasmas
#                    n_init=3,  # Garante que apenas rastreamentos confirmados apareçam
#                    max_cosine_distance=0.2,  # Evita trocas erradas de ID
#                    embedder="mobilenet")

# # Abrir vídeo
# video_path = "video/frangos_inferior-VEED.mp4"
# cap = cv2.VideoCapture(video_path)

# # Configurar saída de vídeo
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# output_path = "resultados/frangos_trackeados_corrigido.avi"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# frame_interval = fps // 5  # Intervalo para pegar 5 frames por segundo
# frame_count = 0
# frame_in_second = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # A cada 1 segundo, processar 5 frames
#     if frame_in_second < 1:
#         # Rodar YOLO no frame
#         results = model.predict(frame, conf=0.7, iou=0.5, imgsz=640)

#         detections = []
#         bounding_boxes = []  # Armazena bounding boxes exatas para desenho

#         for result in results:
#             if result.boxes is not None:
#                 for box in result.boxes.data:
#                     x1, y1, x2, y2, score, class_id = box.cpu().numpy()

#                     if score > 0.3:  # Apenas detecções confiáveis
#                         detections.append(([x1, y1, x2, y2], score, int(class_id)))
#                         bounding_boxes.append((x1, y1, x2, y2))  # Guardar caixas reais

#         # Atualizar rastreamento
#         tracks = tracker.update_tracks(detections, frame=frame)

#         # Desenhar caixas e IDs corretamente
#         for track, bbox in zip(tracks, bounding_boxes):
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             x1, y1, x2, y2 = map(int, bbox)  # Pegamos exatamente as caixas do YOLO

#             # Desenhar bounding box correta
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#             cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         # Salvar frame corrigido
#         out.write(frame)
#         cv2.imshow("Tracking Ajustado", frame)
        
#         frame_in_second += 1
#     else:
#         # Resetar para o próximo segundo
#         if frame_count % frame_interval == 0:
#             frame_in_second = 0
    
#     frame_count += 1

#     # Fechar a janela com a tecla 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar recursos
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Vídeo salvo em: {output_path}")
