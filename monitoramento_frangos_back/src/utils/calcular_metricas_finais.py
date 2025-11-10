import cv2
import numpy as np

def calcular_metricas_finais(pontos_reais, todos_centros, frame_atual, altura_real_px, largura_real_px):
    area_total_m2 = cv2.contourArea(np.array(pontos_reais, dtype=np.float32))
    if len(todos_centros) >= 3:
        pontos = np.array(todos_centros).astype(np.int32)
        hull = cv2.convexHull(pontos)
        mask = np.zeros((altura_real_px, largura_real_px), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        area_detectada_px = np.count_nonzero(mask)
        m2_por_px2 = area_total_m2 / (altura_real_px * largura_real_px)
        area_detectada_m2 = area_detectada_px * m2_por_px2
        return area_total_m2, area_detectada_m2
    return area_total_m2, 0
