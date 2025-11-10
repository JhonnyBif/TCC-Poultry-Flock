import cv2
import numpy as np
import tempfile
import os

def processar_video_memoria(input_path, parte_cima, parte_baixo, fragment= False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Erro ao abrir o vídeo.")

    largura_final, altura_final = 1920, 1080 
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if(fragment) :
        fps_desejado = fps_original
    else :
        fps_desejado = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao_original = total_frames / fps_original
    altura_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices_frames = np.linspace(0, total_frames - 1, int(duracao_original * fps_desejado), dtype=int)

    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps_desejado, (largura_final, altura_final))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in indices_frames:
            altura_corte = 1296
            if(parte_cima) :
                y_inicio = 0
                y_fim = altura_corte
            elif(parte_baixo) :
                y_inicio = altura_original - altura_corte
                y_fim = altura_original
            frame_cortado = frame[y_inicio:y_fim, :]
            frame_redimensionado = cv2.resize(frame_cortado, (largura_final, altura_final))
            out.write(frame_redimensionado)

        frame_count += 1

    cap.release()
    out.release()

    with open(temp_video_path, 'rb') as f:
        video_bytes = f.read()

    os.remove(temp_video_path)
    return video_bytes
