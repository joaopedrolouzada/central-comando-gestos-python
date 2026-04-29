# ==============================================================================
# PROJETO: Central de Comando por Gestos
# FASE 3.1: MOUSE VIRTUAL - OTIMIZAÇÃO MÁXIMA DE FLUIDEZ
# ==============================================================================

import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# ------------------------------------------------------------------------------
# PASSO 1: CONFIGURAÇÃO DE DESEMPENHO (Crucial para fluidez)
# ------------------------------------------------------------------------------

# TÉCNICA 1: Otimizar PyAutoGUI
# Zera o atraso embutido do PyAutoGUI após cada comando. Melhora MUITO a resposta.
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# TÉCNICA 2: Resolução da Câmera
# Ler imagem menor faz o MediaPipe rodar mais rápido.
LARGURA_CAM, ALTURA_CAM = 640, 480 # Resolução padrão otimizada

# Configuração da Webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, LARGURA_CAM) # Define largura
webcam.set(4, ALTURA_CAM) # Define altura

# Configuração MediaPipe Otimizada
mp_maos = mp.solutions.hands
# model_complexity=0: Usa um modelo de IA mais simples e rápido.
maos = mp_maos.Hands(
    model_complexity=0, 
    min_detection_confidence=0.6, # Confiança menor para detecção mais rápida
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------------------------
# PASSO 2: VARIÁVEIS DO MOUSE E CLIQUE (Ajustadas)
# ------------------------------------------------------------------------------

# Tamanho da tela real
LARGURA_TELA, ALTURA_TELA = pyautogui.size()

# Margem de redução (menor para alcançar os cantos mais rápido)
MARGEM_REDUCAO = 70

# Variáveis para Suavização Adaptativa
# coord_anterior: onde o mouse estava
# SUAVIZACAO_BASE: quanto menor, mais rápido o mouse responde.
coord_x_anterior, coord_y_anterior = 0, 0
SUAVIZACAO_BASE = 1.8 

# Variável para o Clique
LIMIAR_CLIQUE = 35 # Distância em pixels na câmera

# Para FPS
tempo_anterior = 0

print("Aperte 'ESC' na janela da câmera para fechar!")

# ------------------------------------------------------------------------------
# PASSO 3: O LOOP INFINITO
# ------------------------------------------------------------------------------

while webcam.isOpened():
    sucesso, frame = webcam.read()
    if not sucesso: continue

    # Efeito espelho
    frame = cv2.flip(frame, 1)
    
    # Converte para RGB
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # O "cérebro" processa (mais rápido agora com model_complexity=0)
    resultado = maos.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for pontos_da_mao in resultado.multi_hand_landmarks:
            # NÃO desenhamos o esqueleto (consome CPU)

            # Ponto 8: Indicador (Mover)
            ponto_8 = pontos_da_mao.landmark[8]
            x_cam_ind = int(ponto_8.x * LARGURA_CAM)
            y_cam_ind = int(ponto_8.y * ALTURA_CAM)

            # Ponto 4: Polegar (Clique)
            ponto_4 = pontos_da_mao.landmark[4]
            x_cam_pol = int(ponto_4.x * LARGURA_CAM)
            y_cam_pol = int(ponto_4.y * ALTURA_CAM)

            # Mapeamento da Câmera para a Tela
            x_tela = np.interp(x_cam_ind, (MARGEM_REDUCAO, LARGURA_CAM - MARGEM_REDUCAO), (0, LARGURA_TELA))
            y_tela = np.interp(y_cam_ind, (MARGEM_REDUCAO, ALTURA_CAM - MARGEM_REDUCAO), (0, ALTURA_TELA))

            # --- SUAVIZAÇÃO ADAPTATIVA (TÉCNICA 3) ---
            # Se o movimento for muito pequeno (trepidação), suavizamos muito.
            # Se o movimento for grande (arrasto), suavizamos pouco para responder rápido.
            
            dist_movimento = math.hypot(x_tela - coord_x_anterior, y_tela - coord_y_anterior)
            
            # Ajuste dinâmico da suavização
            if dist_movimento < 10: 
                suavizacao_atual = SUAVIZACAO_BASE * 3 # Suaviza muito trepidação
            else:
                suavizacao_atual = SUAVIZACAO_BASE # Suaviza normal movimento rápido

            coord_x_atual = coord_x_anterior + (x_tela - coord_x_anterior) / suavizacao_atual
            coord_y_atual = coord_y_anterior + (y_tela - coord_y_anterior) / suavizacao_atual

            # Mover o Mouse Instantaneamente
            pyautogui.moveTo(coord_x_atual, coord_y_atual)
            coord_x_anterior, coord_y_anterior = coord_x_atual, coord_y_atual


            # --- CÓDIGO DO CLIQUE (Otimizado) ---
            # Calcular distância euclidiana
            dist_clique = math.hypot(x_cam_ind - x_cam_pol, y_cam_ind - y_cam_pol)

            if dist_clique < LIMIAR_CLIQUE:
                # Círculo Verde no frame (opcional, consome pouca CPU)
                cv2.circle(frame, (x_cam_ind, y_cam_ind), 10, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
            else:
                # Círculo Magenta normal
                cv2.circle(frame, (x_cam_ind, y_cam_ind), 10, (255, 0, 255), cv2.FILLED)

    # Mostrar FPS (Só para você ver a melhora)
    tempo_atual = time.time()
    fps = 1 / (tempo_atual - tempo_anterior)
    tempo_anterior = tempo_atual
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Mostrar na tela
    cv2.imshow('Mouse Otimizado', frame)

    if cv2.waitKey(1) & 0xFF == 27: break

# Limpeza
webcam.release()
cv2.destroyAllWindows()