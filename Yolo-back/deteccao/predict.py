from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
GITHUB_ASSETS_STEMS
from ultralytics import YOLO, settings
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
source = os.path.normpath(os.path.join(base_dir, '..', 'video', '06_03', 'correndo_recortado.mp4'))
file_name = os.path.splitext(os.path.basename(source))[0]
output_file=  os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', file_name, file_name + '.txt'))

print(nomes_classes)
frames_por_classe = {}
total_deteccoes = {}
if __name__ == '__main__':
  settings.update({'weights_dir': '../weights'})
  settings.update({'datasets_dir': '../datasets'})
  settings.update({'runs_dir': '../runs'})
  weights_path = os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', 'train', 'weights', 'best.pt'))

  model = YOLO(weights_path)
  results_img = model.predict(source=source,  # image or video; single value or a list; URL, PIL (RGB), CV2 (BGR), ...
                          conf=0.25,
                          iou=0.7,  # Non-Maximum Supression (NMS)
                          imgsz=640,
                          show=True,
                          save=True,
                          save_txt=True,  # Save bbox coordenation
                          save_conf=True,  # save_txt must be True
                          save_crop=False,
                          max_det = 1500,  # Maximum number of detections per image
                          show_conf = True, # Show Conf in the image
                          show_labels = True, # Show labels in the image
                          # project='runs/detect',
                          stream=True,  # Do inference now (False) or after (True)
                          line_width=1,  # Bbox line width
                          project=os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect')),  # Local para salvar (diretório)
                          name=file_name
                          )
frame_atual = 0  # Contador de frames analisados
for result in results_img:
    frame_atual += 1  # Atualiza o frame atual

    if result.boxes is not None:
        classes_detectadas = result.boxes.cls.tolist()  # Converte para lista
        
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
            frames_por_classe[nome_classe].add(frame_atual)

fim = time.time()
tempo_execucao = fim - inicio
minutos = int(tempo_execucao // 60)
segundos = int(tempo_execucao % 60)

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
