import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# --- CONFIGURAÇÕES TÉCNICAS ---
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
LARGURA_CAM, ALTURA_CAM = 640, 480
webcam = cv2.VideoCapture(0)
webcam.set(3, LARGURA_CAM)
webcam.set(4, ALTURA_CAM)

mp_maos = mp.solutions.hands
maos = mp_maos.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- VARIÁVEIS DE CONTROLE ---
LARGURA_TELA, ALTURA_TELA = pyautogui.size()
MARGEM_REDUCAO = 80
SUAVIZACAO = 1.8
coord_x_ant, coord_y_ant = 0, 0

# --- CONFIGURAÇÃO TECLADO ---
tamanho_tecla = 80
lista_teclas = [["1", (100, 150)], ["2", (200, 150)], ["3", (300, 150)], ["Limpar", (400, 150)]]
texto_visor = ""

# --- TRAVAS ANTI-CRASH ---
clique_travado = False  # Só deixa clicar se estiver False
tempo_ultimo_clique = 0
COOLDOWN = 0.6 

def desenhar_interface(img, texto):
    # Visor de texto
    cv2.rectangle(img, (100, 50), (550, 120), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, f"TXT: {texto}", (115, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for tecla in lista_teclas:
        nome, pos = tecla
        x, y = pos
        w = tamanho_tecla * 2 if nome == "Limpar" else tamanho_tecla
        cv2.rectangle(img, (x, y), (x + w, y + tamanho_tecla), (255, 0, 255), 2)
        cv2.putText(img, nome, (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

while webcam.isOpened():
    sucesso, frame = webcam.read()
    if not sucesso: break
    frame = cv2.flip(frame, 1)
    desenhar_interface(frame, texto_visor)
    
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = maos.process(imagem_rgb)

    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            # Pontos de interesse
            ind = landmarks.landmark[8]  # Indicador
            pol = landmarks.landmark[4]  # Polegar
            
            x_ind, y_ind = int(ind.x * LARGURA_CAM), int(ind.y * ALTURA_CAM)
            x_pol, y_pol = int(pol.x * LARGURA_CAM), int(pol.y * ALTURA_CAM)
            
            # Distância para clique (Pinça)
            dist = math.hypot(x_ind - x_pol, y_ind - y_pol)

            # --- LÓGICA DO MOUSE (Se não estiver em cima do teclado) ---
            x_t = np.interp(x_ind, (MARGEM_REDUCAO, LARGURA_CAM - MARGEM_REDUCAO), (0, LARGURA_TELA))
            y_t = np.interp(y_ind, (MARGEM_REDUCAO, ALTURA_CAM - MARGEM_REDUCAO), (0, ALTURA_TELA))
            
            curr_x = coord_x_ant + (x_t - coord_x_ant) / SUAVIZACAO
            curr_y = coord_y_ant + (y_t - coord_y_ant) / SUAVIZACAO
            
            pyautogui.moveTo(curr_x, curr_y)
            coord_x_ant, coord_y_ant = curr_x, curr_y

            # --- LÓGICA DO TECLADO ---
            for t in lista_teclas:
                nome, pos = t
                tx, ty = pos
                tw = tamanho_tecla * 2 if nome == "Limpar" else tamanho_tecla
                
                # Se o dedo está na tecla
                if tx < x_ind < tx + tw and ty < y_ind < ty + tamanho_tecla:
                    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + tamanho_tecla), (255, 255, 255), 3)
                    
                    # TENTATIVA DE CLIQUE
                    if dist < 30: # Pinça fechada
                        if not clique_travado and (time.time() - tempo_ultimo_clique) > COOLDOWN:
                            if nome == "Limpar": texto_visor = ""
                            else:
                                texto_visor += nome
                                pyautogui.press(nome)
                            
                            tempo_ultimo_clique = time.time()
                            clique_travado = True # TRAVA
                            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + tamanho_tecla), (0, 255, 0), cv2.FILLED)
                    else:
                        clique_travado = False # Só destrava se abrir a mão

            cv2.circle(frame, (x_ind, y_ind), 10, (255, 0, 255), cv2.FILLED)

    cv2.imshow('Central de Comando v1.0', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

webcam.release()
cv2.destroyAllWindows()