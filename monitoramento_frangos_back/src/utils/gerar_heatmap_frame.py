import numpy as np
import cv2

def gerar_heatmap_frame(frame_base, H_inv, contagem_por_quadrado, frame_atual, passo, altura_real_px, largura_real_px, limite_aglomeracao, contador_aglomeracao):
    heatmap = np.zeros_like(frame_base)
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
                if contador_aglomeracao[chave] >= 5:
                    cv2.fillPoly(heatmap, [quad_original], (0, 0, 255))
                else:
                    porcentagem = min(1.0, media_frame / limite_aglomeracao)
                    if porcentagem <= 0.60:
                        cor = (0, 255, 0)
                    else:
                        esc = (porcentagem - 0.5) / 0.5
                        cor = (0, int((1.0 - esc) * 2 * 225), 255)
                    cv2.fillPoly(heatmap, [quad_original], cor)
    return heatmap