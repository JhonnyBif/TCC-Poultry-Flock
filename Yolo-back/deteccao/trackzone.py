import cv2
import time
import os
import yaml
from ultralytics import solutions

inicio = time.time()

# Configuração inicial
def carregar_nomes_classes(dataset_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, '..', 'yamls', dataset_name+'.yaml')
    yaml_path = os.path.normpath(yaml_path)
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

base_dir = os.path.dirname(os.path.abspath(__file__))
  
yaml_path = 'custom_dataset' 
nomes_classes = carregar_nomes_classes(yaml_path)
source = os.path.normpath(os.path.join(base_dir, '..', 'video', '06_03', 'correndo_recortado.mp4'))
file_name = os.path.splitext(os.path.basename(source))[0]
output_file=  os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', file_name, file_name + '.txt'))
frames_por_classe = {}
total_deteccoes = {"poussin":0}

# Configuração da captura de vídeo
cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Erro ao abrir o vídeo"

# Definir região de interesse
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# Configuração para salvar vídeo de saída
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', file_name, file_name + '_trackzone.avi')), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

weights_path = os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', 'train', 'weights', 'best.pt'))

# Inicializar o TrackZone com o modelo treinado
trackzone = solutions.TrackZone(
    show=True,  # Exibir o vídeo processado
    region=region_points,  # Região de interesse
    model=weights_path,  # Modelo treinado no YOLO11m
    line_width=1,
)

frame_atual = 0

# Processamento do vídeo
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Vídeo finalizado ou erro na leitura do frame.")
        break

    frame_atual += 1
    results = trackzone(im0)
    video_writer.write(results.plot_im)  # Salvar o vídeo processado

    # Processamento das detecções
    if results.total_tracks:
            nome_classe = "poussin"
            total_deteccoes[nome_classe] += results.total_tracks
            frames_por_classe.setdefault(nome_classe, set()).add(frame_atual)

# Finalização do processamento
cap.release()
video_writer.release()
cv2.destroyAllWindows()

fim = time.time()
tempo_execucao = fim - inicio
minutos, segundos = divmod(int(tempo_execucao), 60)

# Salvamento dos resultados
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Total de detecções por classe e média por frame. Tempo de execução: {minutos}m {segundos}s\n")
    for nome_classe, total in total_deteccoes.items():
        num_frames = len(frames_por_classe.get(nome_classe, []))
        media = total / num_frames if num_frames > 0 else 0
        f.write(f"{nome_classe}: {total} detecções em {num_frames} frames -> Média: {media:.2f} detecções/frame\n")

print(f"Resultados salvos em {output_file}")
