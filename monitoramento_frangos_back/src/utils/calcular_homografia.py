import cv2

def calcular_homografia(pontos_imagem, pontos_reais, escala):
    return cv2.getPerspectiveTransform(pontos_imagem, pontos_reais * escala)
