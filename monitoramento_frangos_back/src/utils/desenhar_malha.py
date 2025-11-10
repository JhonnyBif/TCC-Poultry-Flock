import cv2
import numpy as np

def desenhar_malha(frame, H_inv, contagem_por_quadrado, frame_atual, passo, altura_real_px, largura_real_px, limite_aglomeracao, contador_aglomeracao):
    for lin in range(0, altura_real_px, passo):
        for col in range(0, largura_real_px, passo):
            quad_corrigido = np.array([[
                [col, lin], [col + passo, lin],
                [col + passo, lin + passo], [col, lin + passo]
            ]], dtype=np.float32)
            quad_original = cv2.perspectiveTransform(quad_corrigido, H_inv).astype(int)
            chave = (lin // passo, col // passo)
            if chave in contagem_por_quadrado:
                media_frame = contagem_por_quadrado[chave] / frame_atual
                if media_frame > limite_aglomeracao:
                    contador_aglomeracao[chave] += 1
                else:
                    contador_aglomeracao[chave] = 0
                texto = f"{media_frame:.1f}"
                if contador_aglomeracao[chave] >= 5:
                    texto += " AGLOMERACAO"
                pos = tuple(quad_original[0][0])
                cv2.putText(frame, texto, (pos[0]+1, pos[1]+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(frame, texto, (pos[0], pos[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.polylines(frame, [quad_original], True, (0, 255, 0), 1)
            
def desenhar_malha_img(frame, H_inv, contagem_por_quadrado, frame_count, frame_interval, passo, altura_real_px, largura_real_px, limite_aglomeracao):
    for lin in range(0, altura_real_px, passo):
        for col in range(0, largura_real_px, passo):
            quad_corrigido = np.array([[
                [col, lin],
                [col + passo, lin],
                [col + passo, lin + passo],
                [col, lin + passo]
            ]], dtype=np.float32)
            quad_original = cv2.perspectiveTransform(quad_corrigido, H_inv).astype(int)
            chave = (lin // passo, col // passo)
            cv2.polylines(frame, [quad_original], isClosed=True, color=(0, 255, 0), thickness=1)
            if chave in contagem_por_quadrado:
                media_frame = contagem_por_quadrado[chave] / (frame_count // frame_interval + 1)
                pos_texto = tuple(quad_original[0][0])
                texto = f"{media_frame:.1f}"
                if media_frame > limite_aglomeracao:
                    texto += " AGLOMERACAO"
                cv2.putText(frame, texto, (pos_texto[0]+1, pos_texto[1]+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(frame, texto, (pos_texto[0], pos_texto[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def desenhar_malha_trackzone(frame, malha_transformada, contagem_por_quadrado, frame_atual, limite_aglomeracao, contador_aglomeracao):
    for chave, quad_original in malha_transformada.items():
        if chave in contagem_por_quadrado:
            media_frame = contagem_por_quadrado[chave] / frame_atual
            if media_frame > limite_aglomeracao:
                contador_aglomeracao[chave] += 1
            else:
                contador_aglomeracao[chave] = 0
            texto = f"{media_frame:.1f}"
            if contador_aglomeracao[chave] >= 5:
                texto += " AGLOMERACAO"
            pos = tuple(quad_original[0][0])
            cv2.putText(frame, texto, (pos[0]+1, pos[1]+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(frame, texto, (pos[0], pos[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.polylines(frame, [quad_original], True, (0, 255, 0), 1)
