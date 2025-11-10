import cv2

input_video = "../video/31-03/10_minutos"
output_video = input_video + "_recortado_superior.mp4"

def processar_video(input_path, output_path, largura_final=1920, altura_final=1080, fps_desejado=7, inicio="0:01", fim="10:01"):
    cap = cv2.VideoCapture(input_path + '.mp4')
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Obtém FPS e dimensões originais
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    largura_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Converter tempo para frames
    def tempo_para_frames(tempo_str, fps):
        minutos, segundos = map(int, tempo_str.split(":"))
        return int((minutos * 60 + segundos) * fps)

    frame_inicio = tempo_para_frames(inicio, fps_original)
    frame_fim = tempo_para_frames(fim, fps_original)
    
    print(f"Recortando do frame {frame_inicio} até {frame_fim} ({inicio} até {fim})")

    # Criar vídeo de saída com FPS correto
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_desejado, (largura_final, altura_final))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)  # Pula para o frame inicial

    # Calcular quantos frames precisamos para manter a duração correta
    duracao_em_segundos = (frame_fim - frame_inicio) / fps_original
    total_frames_necessarios = int(duracao_em_segundos * fps_desejado)

    print(f"Duração esperada: {duracao_em_segundos:.2f} segundos")
    print(f"Frames necessários para manter FPS correto: {total_frames_necessarios}")

    intervalo_quadros = int(fps_original / fps_desejado)
    frame_count = 0
    frames_processados = 0

    while frames_processados < total_frames_necessarios:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % intervalo_quadros == 0:
            # Cortar apenas a parte superior
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
    print(f"Vídeo salvo como {output_path} com {frames_processados / fps_desejado:.2f} segundos de duração")

processar_video(input_video, output_video, inicio="0:01", fim="10:01")
