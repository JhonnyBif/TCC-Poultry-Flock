import cv2
import os
import shutil
from ultralytics import YOLO
import yaml
import time
import os

inicio = time.time()

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

# Função para garantir que a pasta não seja sobrescrita
def get_unique_output_dir(base_dir):
    counter = 1
    output_dir = base_dir
    while os.path.exists(output_dir):
        output_dir = f"{base_dir}_{counter}"
        counter += 1
    return output_dir

source = os.path.normpath(os.path.join(base_dir, '..', 'video', '06_03', 'correndo_recortado.mp4'))
file_name = os.path.splitext(os.path.basename(source))[0]
output_file=  os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', file_name, file_name + '_img.txt'))

print(nomes_classes)
frames_por_classe = {}
total_deteccoes = {}

weights_path = os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', 'train', 'weights', 'best.pt'))
model = YOLO(weights_path)

# Abre o vídeo
video_title = os.path.basename(source).replace('.mp4', '')
cap = cv2.VideoCapture(source)

# Obtém FPS do vídeo para calcular frames equivalentes a 5 segundos
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = 1 * fps  # Captura um frame a cada 5 segundos

# Define a pasta base para salvar os resultados
base_output_dir = os.path.join(base_dir, '..', 'runs', 'detect', video_title+'_img')
output_dir = get_unique_output_dir(base_output_dir)  # Garante uma pasta única
os.makedirs(output_dir, exist_ok=True)

# Cria a pasta 'labels' dentro da pasta de saída para salvar os arquivos .txt
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se o vídeo terminar
    
    # Processa apenas frames nos intervalos desejados
    if frame_count % frame_interval == 0:
        # Faz a predição no frame
        results = model.predict(source=frame,  
                                conf=0.25,
                                iou=0.7,
                                imgsz=640,
                                save=True,  # Agora o YOLO salva as imagens e txts
                                save_txt=True,
                                save_conf=True,
                                project=os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect')),  
                                name="temp",  # Salva os arquivos em uma pasta temporária
                                exist_ok=True,
                                show=False,
                                max_det = 1500,
                                line_width=1
                                )
        # Verifica se houve detecção
        if results[0].boxes.shape[0] > 0:  
            # Renomeia e move os arquivos de imagem
            temp_dir = os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', 'temp')),
            for file in os.listdir(temp_dir):
                if file.endswith(".jpg"):  # Move apenas imagens
                    src_path = os.path.join(temp_dir, file)
                    dst_filename = f"frame_{frame_count}_" + file  # Evita sobrescritas
                    dst_path = os.path.join(output_dir, dst_filename)
                    shutil.move(src_path, dst_path)

            # Move os arquivos .txt da pasta 'labels' para o diretório final
            labels_temp_dir = os.path.join(base_dir, '..', 'runs', 'detect', "temp", "labels")
            for label_file in os.listdir(labels_temp_dir):
                if label_file.endswith(".txt"):  # Move os arquivos .txt
                    src_path = os.path.join(labels_temp_dir, label_file)
                    dst_filename = f"frame_{frame_count}_" + label_file  # Evita sobrescritas
                    dst_path = os.path.join(labels_dir, dst_filename)
                    shutil.move(src_path, dst_path)
                    
        classes_detectadas = results[0].boxes.cls.tolist()  # Converte para lista
        for classe_id in classes_detectadas:
            classe_id = int(classe_id)
            if 0 <= classe_id < len(nomes_classes):
                nome_classe = nomes_classes[classe_id]  # Pega o nome real
            else:
                nome_classe = f"Classe {classe_id}"  # Caso a classe não exista

            # Atualiza o total de detecções
            if nome_classe not in total_deteccoes:
                total_deteccoes[nome_classe] = 0
            total_deteccoes[nome_classe] += 1

            # Adiciona o frame no qual a classe apareceu
            if nome_classe not in frames_por_classe:
                frames_por_classe[nome_classe] = set()  # Usar set para armazenar apenas frames únicos
            frames_por_classe[nome_classe].add(frame_count)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

fim = time.time()
tempo_execucao = fim - inicio
minutos = int(tempo_execucao // 60)
segundos = int(tempo_execucao % 60)

# Limpa a pasta temporária
shutil.rmtree(os.path.join(base_dir, '..', 'runs', 'detect', "temp"), ignore_errors=True)
with open(output_file, 'w', encoding='utf-8') as f:
    # Escreva o cabeçalho
    print(f"Total de detecções por classe e média por frame e levou {minutos} minutos e {segundos} segundos para executar:")
    f.write(f"Total de detecções por classe e média por frame e levou {minutos} minutos e {segundos} segundos para executar:\n")
    
    # Escreva as detecções por classe
    for nome_classe, total in total_deteccoes.items():
        num_frames = len(frames_por_classe.get(nome_classe, []))
        media = total / num_frames if num_frames > 0 else 0
        
        # Escreva cada linha no arquivo
        print(f"{nome_classe}: {total} detecções em {num_frames} frames -> Média: {media:.2f} detecções/frame")
        f.write(f"{nome_classe}: {total} detecções em {num_frames} frames -> Média: {media:.2f} detecções/frame\n")

print(f"Resultados salvos em {output_file}")
