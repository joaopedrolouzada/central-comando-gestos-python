# ==============================================================================
# PROJETO: Central de Comando por Gestos
# FASE 4: TECLADO HOLOGRÁFICO (Simplificado)
# ==============================================================================

import cv2
import mediapipe as mp
import pyautogui
import time
import math

# ------------------------------------------------------------------------------
# PASSO 1: CONFIGURAR E VARIÁVEIS (Otimizado para desempenho)
# ------------------------------------------------------------------------------
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

LARGURA_CAM, ALTURA_CAM = 640, 480
webcam = cv2.VideoCapture(0)
webcam.set(3, LARGURA_CAM)
webcam.set(4, ALTURA_CAM)

mp_maos = mp.solutions.hands
# model_complexity=0 para velocidade
maos = mp_maos.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- CONFIGURAÇÃO DO TECLADO ---
# Lista de Teclas. Cada item é [Texto, (x_inicial, y_inicial)]
# O tamanho de cada tecla será fixo.
tamanho_tecla = 80
lista_teclas = [
    ["1", (100, 100)],
    ["2", (200, 100)],
    ["3", (300, 100)],
    ["Limpar", (400, 100)], # Tecla maior
]

# Variável para guardar o texto digitado
texto_digitado = ""

# Cooldown para não digitar a mesma tecla 30 vezes por segundo
tempo_ultimo_clique = 0
COOLDOWN_TECLADO = 0.5 # meio segundo de pausa entre cliques

print("Aperte 'ESC' para fechar!")

# ------------------------------------------------------------------------------
# PASSO 2: FUNÇÃO PARA DESENHAR O TECLADO
# ------------------------------------------------------------------------------
def desenhar_teclado(img, teclas, texto):
    # Desenhar o fundo onde o texto aparece
    cv2.rectangle(img, (100, 20), (550, 80), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, texto, (110, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    for tecla in teclas:
        nome, pos = tecla
        x, y = pos
        # Define largura específica para a tecla 'Limpar'
        w = tamanho_tecla * 2 if nome == "Limpar" else tamanho_tecla
        h = tamanho_tecla
        
        # Desenhar o retângulo da tecla
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2) # Magenta Borda
        cv2.putText(img, nome, (x + 10, y + 55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img

# ------------------------------------------------------------------------------
# PASSO 3: O LOOP INFINITO
# ------------------------------------------------------------------------------

while webcam.isOpened():
    sucesso, frame = webcam.read()
    if not sucesso: continue
    frame = cv2.flip(frame, 1) # Espelho

    # A. DESENHAR O TECLADO NA IMAGEM ORIGINAL
    frame = desenhar_teclado(frame, lista_teclas, texto_digitado)

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(imagem_rgb)

    # B. PROCESSAR A MÃO
    if resultado.multi_hand_landmarks:
        for pontos_da_mao in resultado.multi_hand_landmarks:
            # Ponto 8: Indicador (Mover e Clicar)
            ponto_8 = pontos_da_mao.landmark[8]
            x_cam_ind = int(ponto_8.x * LARGURA_CAM)
            y_cam_ind = int(ponto_8.y * ALTURA_CAM)
            
            ponto_4 = pontos_da_mao.landmark[4]
            x_cam_pol = int(ponto_4.x * LARGURA_CAM)
            y_cam_pol = int(ponto_4.y * ALTURA_CAM)

            dist_clique = math.hypot(x_cam_ind - x_cam_pol, y_cam_ind - y_cam_pol)
            # --- O SEGREDO: PROFUNDIDADE Z ---
            # O MediaPipe nos dá o Z normalizado. Quanto menor, mais perto da câmera.
            # Vamos usar um multiplicador para facilitar a visualização no terminal.
            profundidade_z = int(ponto_8.z * -100) # Inverte para Z maior = mais perto
            
            # Limiar de Pressão: Se Z for maior que 10, entendemos como "pressionado"
            limiar_pressao = 10 
            
            # Mostra Z no terminal para você calibrar
          

            # Desenha círculo no dedo indicador (sempre)
            cv2.circle(frame, (x_cam_ind, y_cam_ind), 10, (255, 255, 0), cv2.FILLED)

            # --- VERIFICAR INTERAÇÃO COM AS TECLAS ---
            for tecla in lista_teclas:
                nome, pos = tecla
                x, y = pos
                w = tamanho_tecla * 2 if nome == "Limpar" else tamanho_tecla
                h = tamanho_tecla

                # 1. Verificar se o dedo está DENTRO da caixa da tecla
                if x < x_cam_ind < x + w and y < y_cam_ind < y + h:
                    # Se está dentro, muda a borda para BRANCO (pré-clique)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 4)

                    # 2. Verificar se houve "pressão" (Z maior que limiar)
                   # Se a distância entre polegar e indicador for pequena, clicou!
                    if dist_clique < 30:
                        
                        # 3. Verificar Cooldown (para não clicar sem parar)
                        if (time.time() - tempo_ultimo_clique) > COOLDOWN_TECLADO:
                            
                            # Muda a tecla para VERDE (clique confirmado)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            
                            # AÇÃO:
                            if nome == "Limpar":
                                texto_digitado = ""
                            else:
                                texto_digitado += nome
                                # Pressiona a tecla real no computador
                                pyautogui.press(nome)
                            
                            # Atualiza o tempo do último clique
                            tempo_ultimo_clique = time.time()


    # Mostrar na tela
    cv2.imshow('Teclado Holografico - Fase 4', frame)

    if cv2.waitKey(1) & 0xFF == 27: break

# Limpeza
webcam.release()
cv2.destroyAllWindows()