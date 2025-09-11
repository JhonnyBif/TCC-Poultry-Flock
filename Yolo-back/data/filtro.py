import os

# Caminho para a pasta de anotações do seu dataset com 3 classes (dataset2)
dataset2_labels_dir = "./data/personalizado/test/labels"
dataset2_images_dir = "C./data/personalizado/test/labels/images"

# Loop para percorrer todos os arquivos .txt do dataset2
for filename in os.listdir(dataset2_labels_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(dataset2_labels_dir, filename)

        # Abrir o arquivo de anotação
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Filtrar e alterar as anotações
        new_lines = []
        delete_file = False
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # ID da classe
            
            # Se a classe for "pousin" (classe 2), substitua por 0
            if class_id == 2:
                parts[0] = "0"  # Substitui a classe 2 por 0
                new_lines.append(" ".join(parts) + "\n")
            else:
                delete_file = True  # Marque para deletar o arquivo se a classe não for pousin

        # Se houver anotações da classe "pousin", escreva no arquivo
        if new_lines:
            with open(file_path, "w") as file:
                file.writelines(new_lines)
        else:
            # Se não houver anotações da classe "pousin", exclua o arquivo .txt
            os.remove(file_path)
            
            # Remover a imagem correspondente ao arquivo .txt
            image_path = os.path.join(dataset2_images_dir, filename.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                os.remove(image_path)

print("Processamento concluído!")

