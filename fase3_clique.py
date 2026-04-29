# ==============================================================================
# PROJETO: Central de Comando por Gestos
# FASE 3: O CLIQUE VIRTUAL (Gesto de Pinça)
# ==============================================================================

import cv2
import mediapipe as mp
import pyautogui
import time
import math # BIBLIOTECA 5: Matemática. Para calcular a distância.

# ------------------------------------------------------------------------------
# PASSO 1: CONFIGURAR AS BIBLIOTECAS E VARIÁVEIS (Igual Fase 2)
# ------------------------------------------------------------------------------

# Configuração MediaPipe
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
# Aumentei a confiança de detecção para 0.85 para o clique ser mais preciso
maos = mp_maos.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5)

# Configuração Webcam
webcam = cv2.VideoCapture(0)

# Variáveis para FPS
tempo_anterior = 0
tempo_atual = 0

# Variáveis do Mouse (Igual Fase 2)
LARGURA_TELA, ALTURA_TELA = pyautogui.size()
MARGEM_REDUCAO = 100
coord_x_anterior, coord_y_anterior = 0, 0
coord_x_atual, coord_y_atual = 0, 0
SUAVIZACAO = 2   # Reduzi para 2 para o mouse ser mais rápido

# --- NOVA VARIÁVEL PARA O CLIQUE ---
# Limiar de distância: Se a distância entre os dedos for menor que isso, é um clique.
# Esse valor depende da sua câmera. Se estiver difícil clicar, aumente para 35. Se clicar sozinho, diminua para 25.
LIMIAR_CLIQUE = 30
# -------------------------------------

print("Aperte 'ESC' para fechar!")
pyautogui.FAILSAFE = False

# ------------------------------------------------------------------------------
# PASSO 2: O LOOP INFINITO
# ------------------------------------------------------------------------------

while webcam.isOpened():
    sucesso, frame = webcam.read()
    if not sucesso:
        print("Ignorando frame vazio.")
        continue

    # Ajustes da imagem
    frame = cv2.flip(frame, 1)
    altura_camera, largura_camera, _ = frame.shape
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # O "cérebro" processa
    resultado = maos.process(imagem_rgb)

    # Verificamos se alguma mão foi detectada
    if resultado.multi_hand_landmarks:
        for pontos_da_mao in resultado.multi_hand_landmarks:
            # Desenha o esqueleto
            mp_desenho.draw_landmarks(frame, pontos_da_mao, mp_maos.HAND_CONNECTIONS)

            # --- EXTRAIR COORDENADAS DO DEDO INDICADOR (Igual Fase 2) ---
            # Ponto 8: Indicador Tip
            ponto_8_indicador = pontos_da_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP]
            x_camera_ind = int(ponto_8_indicador.x * largura_camera)
            y_camera_ind = int(ponto_8_indicador.y * altura_camera)

            # --- NOVO: EXTRAIR COORDENADAS DO POLEGAR ---
            # Ponto 4: Thumb Tip (Ponta do Polegar)
            ponto_4_polegar = pontos_da_mao.landmark[mp_maos.HandLandmark.THUMB_TIP]
            x_camera_pol = int(ponto_4_polegar.x * largura_camera)
            y_camera_pol = int(ponto_4_polegar.y * altura_camera)


            # --- MAPEAMENTO E SUAVIZAÇÃO (Igual Fase 2 - Baseado no Indicador) ---
            import numpy as np
            x_tela = np.interp(x_camera_ind, (MARGEM_REDUCAO, largura_camera - MARGEM_REDUCAO), (0, LARGURA_TELA))
            y_tela = np.interp(y_camera_ind, (MARGEM_REDUCAO, altura_camera - MARGEM_REDUCAO), (0, ALTURA_TELA))

            coord_x_atual = coord_x_anterior + (x_tela - coord_x_anterior) / SUAVIZACAO
            coord_y_atual = coord_y_anterior + (y_tela - coord_y_anterior) / SUAVIZACAO

            # Mover o Mouse
            pyautogui.moveTo(coord_x_atual, coord_y_atual, _pause=False)
            coord_x_anterior, coord_y_anterior = coord_x_atual, coord_y_atual


            # --------------------------------------------------------------------------
            # PASSO 4: O CÓDIGO DO CLIQUE
            # --------------------------------------------------------------------------
            
            # 1. Calcular a distância euclidiana entre a ponta do indicador (8) e o polegar (4)
            # É a fórmula da hipotenusa: raiz((x2-x1)^2 + (y2-y1)^2)
            distancia = math.hypot(x_camera_ind - x_camera_pol, y_camera_ind - y_camera_pol)

            # 2. Verificar se a distância é menor que o limiar 
            if distancia < LIMIAR_CLIQUE:
                # Desenha o círculo do indicador em VERDE para mostrar que clicou
                cv2.circle(frame, (x_camera_ind, y_camera_ind), 15, (0, 255, 0), cv2.FILLED)
                
                # Executa o clique de verdade. O '_pause=False' é para não travar o loop.
                pyautogui.click(_pause=False)
                
                # Pequena pausa para não clicar 30 vezes por segundo (simula um clique normal)
                #time.sleep(0.1) 
            else:
                # Se não clicou, desenha o círculo em MAGENTA (normal)
                cv2.circle(frame, (x_camera_ind, y_camera_ind), 15, (255, 0, 255), cv2.FILLED)
            
            # Opcional: Desenhar círculo no polegar também
            cv2.circle(frame, (x_camera_pol, y_camera_pol), 15, (255, 255, 0), cv2.FILLED)

    # --------------------------------------------------------------------------

    # Mostrar FPS (opcional)
    tempo_atual = time.time()
    fps = 1 / (tempo_atual - tempo_anterior)
    tempo_anterior = tempo_atual
    #cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Mostrar na tela
    cv2.imshow('Clique Virtual - Fase 3', frame)

    # Fechar com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Limpeza
webcam.release()
cv2.destroyAllWindows()
print("Programa encerrado.")