# ==============================================================================
# PROJETO: Central de Comando por Gestos
# FASE 1: O "OLHO" BÁSICO (Detectar Mão e Desenhar Pontos)
# ==============================================================================

import cv2          # BIBLIOTECA 1: OpenCV. É o "olho". Controla a webcam.
import mediapipe as mp # BIBLIOTECA 2: MediaPipe do Google. É o "cérebro" de IA.

# ------------------------------------------------------------------------------
# PASSO 1: CONFIGURAR AS BIBLIOTECAS (Ativar os superpoderes)
# ------------------------------------------------------------------------------

# Ativamos o módulo de soluções de mãos do MediaPipe
mp_maos = mp.solutions.hands

# Ativamos a ferramenta de desenho do MediaPipe (para ver os pontos na tela)
mp_desenho = mp.solutions.drawing_utils

# Criamos o nosso detector de mãos "cérebro"
# min_detection_confidence=0.7: O programa precisa ter 70% de certeza que é uma mão
# min_tracking_confidence=0.5: O programa precisa ter 50% de certeza para rastrear
maos = mp_maos.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)


# ------------------------------------------------------------------------------
# PASSO 2: INICIAR A WEBCAM (O "olho" abre)
# ------------------------------------------------------------------------------

# cv2.VideoCapture(0): Tenta ligar a câmera padrão (0). Se tiver externa, tente 1.
webcam = cv2.VideoCapture(0)

print("Aperte 'ESC' para fechar a janela!")


# ------------------------------------------------------------------------------
# PASSO 3: O LOOP INFINITO (O programa roda enquanto a câmera estiver ligada)
# ------------------------------------------------------------------------------

while webcam.isOpened():
    # A. Lê o frame atual da webcam
    # 'sucesso' é True/False se a câmera funcionou. 'frame' é a imagem real.
    sucesso, frame = webcam.read()

    # Se a câmera falhar por um frame, pula para o próximo (não quebra o programa)
    if not sucesso:
        print("Ignorando frame vazio da webcam.")
        continue

    # B. AJUSTES DA IMAGEM (O MediaPipe é exigente!)
    # 1. Inverte a imagem horizontalmente (efeito espelho), é mais natural para nós.
    frame = cv2.flip(frame, 1)

    # 2. Converte a cor de BGR (padrão do OpenCV) para RGB (padrão que o MediaPipe entende)
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # C. O "CÉREBRO" PROCESSA A IMAGEM
    # O MediaPipe analisa a imagem RGB e procura mãos.
    resultado = maos.process(imagem_rgb)


    # D. O DESENHO (Visualizar o que o cérebro achou)
    # Verificamos se o "cérebro" realmente achou alguma mão
    if resultado.multi_hand_landmarks:
        # Se achou, ele percorre CADA mão detectada (pode ser mais de uma)
        for pontos_da_mao in resultado.multi_hand_landmarks:
            # mp_desenho: Usa a ferramenta de desenho para colocar na imagem original (frame)
            # pontos_da_mao: As coordenadas dos 21 pontos
            # mp_maos.HAND_CONNECTIONS: Diz para desenhar as linhas conectando os pontos
            mp_desenho.draw_landmarks(frame, pontos_da_mao, mp_maos.HAND_CONNECTIONS)


    # E. MOSTRAR NA TELA
    # cv2.imshow: Abre uma janela com o nome "Olho do Projeto" e mostra o frame com os desenhos
    cv2.imshow('Olho do Projeto - Fase 1', frame)


    # F. CÓDIGO PARA FECHAR (Essencial!)
    # Espera 1 milissegundo (cv2.waitKey(1)). Se a tecla 'ESC' (código 27) for apertada, sai do loop.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------------------------------------------------------------------
# PASSO 4: LIMPEZA (Fechar tudo ao sair)
# ------------------------------------------------------------------------------
webcam.release()       # Desliga a webcam (apaga a luzinha)
cv2.destroyAllWindows() # Fecha todas as janelas do OpenCV
print("Programa encerrado com sucesso.")