# ==============================================================================
# PROJETO: Central de Comando por Gestos
# FASE 2: MOUSE VIRTUAL (Mover o Cursor com Suavização)
# ==============================================================================

import cv2
import mediapipe as mp
import pyautogui    # BIBLIOTECA 3: Automação. Move o mouse de verdade.
import time         # BIBLIOTECA 4: Tempo. Para contar FPS.

# ------------------------------------------------------------------------------
# PASSO 1: CONFIGURAR AS BIBLIOTECAS E VARIÁVEIS
# ------------------------------------------------------------------------------

# Configuração básica do MediaPipe (igual à Fase 1)
mp_maos = mp.solutions.hands
#mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Configuração da Webcam (0 é padrão)
webcam = cv2.VideoCapture(0)

# Variáveis para FPS (frames por segundo - para ver se está rápido)
tempo_anterior = 0
tempo_atual = 0

# --- NOVAS VARIÁVEIS PARA O MOUSE ---
# Pega o tamanho real da sua tela (ex: 1920x1080)
LARGURA_TELA, ALTURA_TELA = pyautogui.size()

# Margem de redução: Define uma "área ativa" menor na câmera para alcançar os cantos da tela facilmente
MARGEM_REDUCAO = 100

# Coordenadas anteriores e atuais para suavização (para não pular)
coord_x_anterior, coord_y_anterior = 0, 0
coord_x_atual, coord_y_atual = 0, 0

# Fator de suavização (quanto maior, mais lento/suave o mouse)
SUAVIZACAO = 5
# -------------------------------------

print("Aperte 'ESC' para fechar!")

# Desativa a falha de segurança do PyAutoGUI (senão ele para se o mouse for pro canto)
pyautogui.FAILSAFE = False

# ------------------------------------------------------------------------------
# PASSO 2: O LOOP INFINITO
# ------------------------------------------------------------------------------

while webcam.isOpened():
    sucesso, frame = webcam.read()
    if not sucesso:
        print("Ignorando frame vazio.")
        continue

    # Ajustes da imagem (efeito espelho e cor RGB)
    frame = cv2.flip(frame, 1)
    # Pega o tamanho da imagem da câmera (normalmente 640x480)
    altura_camera, largura_camera, _ = frame.shape

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # O "cérebro" processa
    resultado = maos.process(imagem_rgb)

    # --------------------------------------------------------------------------
    # PASSO 3: O NOVO CÓDIGO DO MOUSE
    # --------------------------------------------------------------------------
    
    # Verificamos se alguma mão foi detectada
    if resultado.multi_hand_landmarks:
        for pontos_da_mao in resultado.multi_hand_landmarks:
            # Desenha o esqueleto (básico da Fase 1)
            mp_desenho.draw_landmarks(frame, pontos_da_mao, mp_maos.HAND_CONNECTIONS)

            # --- EXTRAIR COORDENADAS DO DEDO INDICADOR ---
            # O MediaPipe nos dá 21 pontos. O ponto 8 é a ponta do indicador.
            # mp_maos.HandLandmark.INDEX_FINGER_TIP é igual a 8.
            ponto_8_indicador = pontos_da_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP]
            
            # As coordenadas X e Y do MediaPipe são normalizadas (0.0 a 1.0).
            # Precisamos multiplicar pela largura/altura da CÂMERA para ter pixels reais.
            x_camera = int(ponto_8_indicador.x * largura_camera)
            y_camera = int(ponto_8_indicador.y * altura_camera)

            # --- MAPEAMENTO DA CÂMERA PARA A TELA ---
            # Usamos a função 'interp' do NumPy para converter.
            # Se o dedo estiver na margem da câmera, o mouse vai pro canto da tela.
            import numpy as np
            
            x_tela = np.interp(x_camera, (MARGEM_REDUCAO, largura_camera - MARGEM_REDUCAO), (0, LARGURA_TELA))
            y_tela = np.interp(y_camera, (MARGEM_REDUCAO, altura_camera - MARGEM_REDUCAO), (0, ALTURA_TELA))

            # --- APLICAR SUAVIZAÇÃO (Resolvendo o "Pulo") ---
            # coord_x_atual = onde o dedo está AGORA na tela
            # coord_x_anterior = onde o mouse estava no frame PASSADO
            coord_x_atual = coord_x_anterior + (x_tela - coord_x_anterior) / SUAVIZACAO
            coord_y_atual = coord_y_anterior + (y_tela - coord_y_anterior) / SUAVIZACAO

            # --- MOVER O MOUSE DE VERDADE ---
            # Usamos PyAutoGUI para mover o cursor para as novas coordenadas suavizadas
            # '_pause=False' é essencial para não travar o loop
            pyautogui.moveTo(coord_x_atual, coord_y_atual, _pause=False)
            
            # Atualiza as coordenadas anteriores para o próximo frame
            coord_x_anterior, coord_y_anterior = coord_x_atual, coord_y_atual

            # Opcional: Desenhar um círculo no dedo que move o mouse
            cv2.circle(frame, (x_camera, y_camera), 15, (255, 0, 255), cv2.FILLED)

    # --------------------------------------------------------------------------

    # Mostrar FPS na tela (opcional, só para controle)
    tempo_atual = time.time()
    fps = 1 / (tempo_atual - tempo_anterior)
    tempo_anterior = tempo_atual
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Mostrar na tela
    cv2.imshow('Mouse Virtual - Fase 2', frame)

    # Fechar com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Limpeza
webcam.release()
cv2.destroyAllWindows()
print("Programa encerrado.")