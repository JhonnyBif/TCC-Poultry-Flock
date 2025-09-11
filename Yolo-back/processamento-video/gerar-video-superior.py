import cv2
import numpy as np

input_video = "../video/06_03/superior"
output_video = input_video + "_processado.mp4"
print("Começou")

def processar_video(input_path, output_path, largura_final=1920, altura_final=1080, fps_desejado=7):
    cap = cv2.VideoCapture(input_path + '.mp4')
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Obtém FPS e dimensões originais
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao_original = total_frames / fps_original
    largura_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS original: {fps_original:.2f}, Duração original: {duracao_original:.2f} segundos")

    # Criar vídeo de saída com FPS correto
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_desejado, (largura_final, altura_final))

    # Calcular quais frames selecionar para manter a duração correta
    indices_frames = np.linspace(0, total_frames - 1, int(duracao_original * fps_desejado), dtype=int)

    print(f"Processando exatamente {len(indices_frames)} frames para manter a duração original.")

    frame_count = 0
    frames_processados = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in indices_frames:
            # Cortar a parte inferior do vídeo
            altura_corte = 1296
            y_inicio = 0  # Pegamos desde o topo
            y_fim = altura_corte  # Pegamos os primeiros 1280 pixels
            frame_cortado = frame[y_inicio:y_fim, :]

            # Redimensionar para 1280x768
            frame_redimensionado = cv2.resize(frame_cortado, (largura_final, altura_final))

            # Salvar no novo vídeo
            out.write(frame_redimensionado)
            frames_processados += 1

        frame_count += 1  # Avança para o próximo quadro
    
    cap.release()
    out.release()
    print(f"Vídeo salvo como {output_path} com duração {frames_processados / fps_desejado:.2f} segundos (mesma duração do original).")

processar_video(input_video, output_video)
