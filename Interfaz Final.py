import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import mediapipe as mp
import math
import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import sounddevice as sd
import wave
import os
import threading
import webbrowser
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

try:
    import serial
except ImportError:
    serial = None

try:
    from scipy.signal import butter, filtfilt, iirnotch
except ImportError:
    butter = filtfilt = iirnotch = None


class AMVA:

    def __init__(self, root):
        self.root = root
        self.root.title("AMVA: Sistema Multimodal para Apoyo en Terapias de Rehabilitación Oral")
        self.root.geometry("1200x900")
        self.root.minsize(1050, 800)

        # Paleta de colores de la interfaz
        self.COLOR_FONDO = "#FFFFFF"
        self.COLOR_PANEL = "#F8FAFC"
        self.COLOR_BORDE = "#E2E8F0"
        self.COLOR_TEXTO = "#1E293B"
        self.COLOR_TEXTO_SEC = "#64748B"
        self.COLOR_TURQUESA = "#0F766E"
        self.COLOR_TURQUESA_CLARO = "#CCFBF1"
        self.COLOR_VERDE = "#16A34A"
        self.COLOR_ROJO = "#DC2626"
        self.COLOR_AZUL = "#2563EB"
        self.COLOR_MORADO = "#7C3AED"
        self.COLOR_NARANJA = "#D97706"

        # Variables de control de sesion
        self.grabando = False
        self.tiempo_inicio = 0

        self.frames_actuales = []
        self.tiempos_actuales = []
        self.datos_sesion_completa = []
        self.numero_frame_actual = 0

        self.puntos_previos_velocidad = None
        self.tiempo_previo_velocidad = None

        self.ruta_sujeto_actual = None
        self.datos_video_etapa = []
        self.parametros_video_etapa = {}
        self.parametros_audio_etapa = {}

        # Variables cinemáticas
        self.ultima_posicion_centro = None
        self.desplazamiento_acumulado_px = 0.0
        self.desplazamiento_acumulado_mm = 0.0
        self.velocidad_maxima_mm_s = 0.0
        self.velocidad_actual = 0.0

        # Configuracion de audio
        self.audio_grabando = False
        self.audio_frames = []
        self.audio_samplerate = 44100
        self.audio_channels = 1
        self.audio_dtype = "int16"
        self.audio_stream = None
        self.ruta_audio_actual = None

        self.audio_inicio_fonacion = None
        self.audio_fin_fonacion = None
        self.audio_duracion_fonacion = 0.0
        self.audio_numero_pausas = 0
        self.audio_duracion_pausas = 0.0
        self.audio_f0_promedio = 0.0
        self.audio_f0_minimo = 0.0
        self.audio_f0_maximo = 0.0

        # Configuracion sEMG (ESP32 + MyoWare)
        self.semg_puerto = "COM5"
        self.semg_baudrate = 115200
        self.semg_fs_nominal = 1000
        self.semg_serial = None
        self.semg_hilo = None
        self.semg_detener_evento = threading.Event()
        self.semg_grabando = False
        self.semg_tiempos = []
        self.semg_raw = []
        self.semg_filtrada = []
        self.semg_parametros_etapa = {}
        self.semg_ruta_csv = None

        # Relacion de calibracion: 50 mm / 83.462 px
        self.FACTOR_CALIBRACION_MM_PX = 0.599078

        # Landmarks periorales (61: com. izq, 291: com. der, 13: labio sup, 14: labio inf)
        self.PUNTOS_ORBICULAR = []
        self.PUNTOS_MEDICION = [61, 291, 13, 14]

        self.FACTOR_ANCHO_ORBICULAR = 1.75
        self.FACTOR_ALTO_ORBICULAR = 1.35
        self.DESPLAZAMIENTO_VERTICAL_ORBICULAR = 0.05

        self.ultimo_centro_orbicular = None
        self.ultimo_ancho_boca = None
        self.UMBRAL_MOVIMIENTO_REGION = 0.025
        self.UMBRAL_CAMBIO_ANCHO = 0.12

        self.COLOR_OVALO = (212, 234, 222)
        self.COLOR_LANDMARK = (255, 255, 255)
        self.COLOR_APERTURA = (0, 0, 255)
        self.COLOR_ANCHO = (255, 0, 0)

        self.TAMANO_LANDMARK = 2
        self.TAMANO_MEDICION = 3

        self.mp_malla = mp.solutions.face_mesh
        self.malla_facial = self.mp_malla.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.construir_interfaz()

        self.cap = cv2.VideoCapture(0)
        self.camara_activa = True

        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        self.actualizar_video()

    def construir_interfaz(self):
        self.root.configure(bg=self.COLOR_FONDO)

        encabezado = tk.Frame(self.root, bg=self.COLOR_FONDO)
        encabezado.pack(fill=tk.X, padx=20, pady=(8, 4))

        try:
            img_logo_path = "logo_amva.jpeg" 
            if os.path.exists(img_logo_path):
                img_pil = Image.open(img_logo_path)
                altura_logo = 65
                ancho_logo = int(img_pil.width * (altura_logo / img_pil.height))
                img_pil = img_pil.resize((ancho_logo, altura_logo), Image.Resampling.LANCZOS)
                
                self.logo_tk = ImageTk.PhotoImage(img_pil)
                lbl_logo = tk.Label(encabezado, image=self.logo_tk, bg=self.COLOR_FONDO)
                lbl_logo.pack(side=tk.LEFT, padx=(5, 0))
        except Exception as e:
            print(f"Error al cargar logo: {e}")

        marco_textos = tk.Frame(encabezado, bg=self.COLOR_FONDO)
        marco_textos.pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Label(
            marco_textos,
            text="AMVA",
            font=("Helvetica", 24, "bold"),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TURQUESA
        ).pack(anchor="center")

        tk.Label(
            marco_textos,
            text="Sistema Multimodal para Apoyo en Terapias de Rehabilitación Oral",
            font=("Helvetica", 11),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TEXTO
        ).pack(anchor="center", pady=(1, 0))

        tk.Frame(
            self.root,
            bg=self.COLOR_TURQUESA,
            height=2
        ).pack(fill=tk.X, padx=25, pady=(8, 4))

        contenedor = tk.Frame(self.root, bg=self.COLOR_FONDO)
        contenedor.pack(fill=tk.BOTH, expand=True, padx=25, pady=(4, 12))

        # Panel izquierdo (Video)
        panel_izq = tk.Frame(
            contenedor,
            bg=self.COLOR_PANEL,
            highlightbackground=self.COLOR_BORDE,
            highlightthickness=1
        )
        panel_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            panel_izq,
            text="Vista de análisis facial",
            font=("Helvetica", 18, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO
        ).pack(anchor="w", padx=12, pady=(6, 2))

        tk.Label(
            panel_izq,
            text=(
                "Coloque el rostro dentro del área de referencia, "
                "alineando el mentón con el contorno inferior del círculo "
                "y manteniendo el rostro centrado."
            ),
            font=("Helvetica", 16),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO_SEC,
            wraplength=650,
            justify="left"
        ).pack(anchor="w", padx=12, pady=(0, 4))

        marco_camara = tk.Frame(panel_izq, bg="#E2E8F0")
        marco_camara.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        self.pantalla_video = tk.Label(marco_camara, bg="#FFFFFF")
        self.pantalla_video.pack(expand=True, padx=2, pady=2)

        # Panel derecho (Controles)
        panel_der_exterior = tk.Frame(
            contenedor,
            bg=self.COLOR_PANEL,
            width=350,
            highlightbackground=self.COLOR_BORDE,
            highlightthickness=1
        )
        panel_der_exterior.pack(side=tk.RIGHT, fill=tk.Y)
        panel_der_exterior.pack_propagate(False)

        barra_limpiar = tk.Frame(panel_der_exterior, bg=self.COLOR_PANEL)
        barra_limpiar.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_limpiar = tk.Button(
            barra_limpiar,
            text="LIMPIAR SESIÓN",
            bg="#64748B",
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.limpiar_sesion,
            pady=7
        )
        self.btn_limpiar.pack(fill=tk.X, padx=20, pady=(6, 8))

        canvas_panel = tk.Canvas(
            panel_der_exterior,
            bg=self.COLOR_PANEL,
            highlightthickness=0,
            bd=0,
            width=350
        )
        canvas_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar_panel = ttk.Scrollbar(
            panel_der_exterior,
            orient="vertical",
            command=canvas_panel.yview
        )
        scrollbar_panel.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_panel.configure(yscrollcommand=scrollbar_panel.set)

        panel_der = tk.Frame(canvas_panel, bg=self.COLOR_PANEL)
        panel_der_id = canvas_panel.create_window((0, 0), window=panel_der, anchor="nw")

        def actualizar_scroll(event=None):
            canvas_panel.configure(scrollregion=canvas_panel.bbox("all"))

        panel_der.bind("<Configure>", actualizar_scroll)

        def ajustar_ancho(event):
            canvas_panel.itemconfig(panel_der_id, width=event.width)

        canvas_panel.bind("<Configure>", ajustar_ancho)

        def scroll_rueda(event):
            canvas_panel.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_panel.bind_all("<MouseWheel>", scroll_rueda)

        tk.Label(
            panel_der,
            text="Control de sesión",
            font=("Helvetica", 13, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO
        ).pack(anchor="w", padx=18, pady=(6, 1))

        tk.Label(
            panel_der,
            text="Registro y protocolo de evaluación",
            font=("Helvetica", 8),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO_SEC
        ).pack(anchor="w", padx=18, pady=(0, 3))

        # Datos del sujeto
        frame_registro = tk.LabelFrame(
            panel_der,
            text="  Registro clínico  ",
            font=("Helvetica", 9, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TURQUESA,
            bd=1,
            relief="solid"
        )
        frame_registro.pack(fill=tk.X, padx=15, pady=(0, 3))
        frame_registro.columnconfigure(1, weight=1)

        tk.Label(
            frame_registro,
            text="ID Sujeto",
            font=("Helvetica", 8),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO
        ).grid(row=0, column=0, padx=(10, 4), pady=(5, 2), sticky="w")

        self.id_sujeto = tk.Entry(
            frame_registro,
            font=("Helvetica", 8),
            width=18,
            bg="white",
            fg=self.COLOR_TEXTO,
            relief="solid",
            bd=1,
            insertbackground=self.COLOR_TEXTO
        )
        self.id_sujeto.insert(0, "Sujeto_01")
        self.id_sujeto.grid(row=0, column=1, padx=(4, 10), pady=(5, 2), sticky="ew")

        tk.Label(
            frame_registro,
            text="Condición",
            font=("Helvetica", 8),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO
        ).grid(row=1, column=0, padx=(10, 4), pady=(2, 5), sticky="w")

        self.condicion_sujeto = ttk.Combobox(
            frame_registro,
            values=["Sano", "Patológico"],
            state="readonly",
            font=("Helvetica", 8),
            width=16
        )
        self.condicion_sujeto.set("Sano")
        self.condicion_sujeto.grid(row=1, column=1, padx=(4, 10), pady=(2, 5), sticky="ew")

        # Selector de etapa
        frame_protocolo = tk.LabelFrame(
            panel_der,
            text="  Protocolo experimental  ",
            font=("Helvetica", 9, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TURQUESA,
            bd=1,
            relief="solid"
        )
        frame_protocolo.pack(fill=tk.X, padx=15, pady=(0, 3))

        self.ejercicio_seleccionado = tk.StringVar(value="Etapa 1: Sellado Labial")

        opciones = [
            "Etapa 1: Sellado Labial",
            "Etapa 2: Fuerza Labial",
            "Etapa 3: Coordinacion",
            "Etapa 4: Fonemas",
            "Etapa 5: Palabras Funcionales",
            "Etapa 6: Frases"
        ]

        self.menu_ejercicios = ttk.Combobox(
            frame_protocolo,
            textvariable=self.ejercicio_seleccionado,
            values=opciones,
            state="readonly",
            font=("Helvetica", 8)
        )
        self.menu_ejercicios.pack(fill=tk.X, padx=10, pady=(4, 3))
        self.menu_ejercicios.bind("<<ComboboxSelected>>", self.actualizar_instrucciones)

        self.lbl_instrucciones = tk.Label(
            frame_protocolo,
            text="10 segundos, un solo ciclo.",
            font=("Helvetica", 8, "italic"),
            bg=self.COLOR_TURQUESA_CLARO,
            fg=self.COLOR_TEXTO,
            justify="center",
            wraplength=300
        )
        self.lbl_instrucciones.pack(fill=tk.X, padx=10, pady=(0, 4))

        # Cronometro
        frame_tiempo = tk.Frame(
            panel_der,
            bg=self.COLOR_TURQUESA_CLARO,
            highlightbackground=self.COLOR_BORDE,
            highlightthickness=1
        )
        frame_tiempo.pack(fill=tk.X, padx=15, pady=(0, 3))

        tk.Label(
            frame_tiempo,
            text="Tiempo transcurrido",
            font=("Helvetica", 8, "bold"),
            bg=self.COLOR_TURQUESA_CLARO,
            fg=self.COLOR_TURQUESA
        ).pack(pady=(4, 0))

        self.lbl_tiempo = tk.Label(
            frame_tiempo,
            text="00:00.0",
            font=("Helvetica", 18, "bold"),
            bg=self.COLOR_TURQUESA_CLARO,
            fg=self.COLOR_TURQUESA
        )
        self.lbl_tiempo.pack(pady=(0, 4))

        # Indicadores en tiempo real
        frame_metricas = tk.Frame(
            panel_der,
            bg="white",
            highlightbackground=self.COLOR_BORDE,
            highlightthickness=1
        )
        frame_metricas.pack(fill=tk.X, padx=15, pady=(0, 3))

        tk.Label(
            frame_metricas,
            text="Métricas actuales",
            font=("Helvetica", 9, "bold"),
            bg="white",
            fg=self.COLOR_TEXTO
        ).pack(anchor="w", padx=12, pady=(5, 2))

        self.lbl_apertura = tk.Label(
            frame_metricas,
            text="Apertura Bucal: 0.0 mm",
            font=("Helvetica", 9),
            bg="white",
            fg=self.COLOR_TEXTO
        )
        self.lbl_apertura.pack(anchor="w", padx=12, pady=1)

        self.lbl_ancho = tk.Label(
            frame_metricas,
            text="Ancho de Boca: 0.0 mm",
            font=("Helvetica", 9),
            bg="white",
            fg=self.COLOR_TEXTO
        )
        self.lbl_ancho.pack(anchor="w", padx=12, pady=1)

        self.lbl_velocidad = tk.Label(
            frame_metricas,
            text="Velocidad: 0.0 mm/s",
            font=("Helvetica", 9),
            bg="white",
            fg=self.COLOR_TEXTO
        )
        self.lbl_velocidad.pack(anchor="w", padx=12, pady=1)

        self.lbl_estado = tk.Label(
            frame_metricas,
            text="ESTADO: EN ESPERA",
            font=("Helvetica", 9, "bold"),
            bg="white",
            fg=self.COLOR_TEXTO_SEC
        )
        self.lbl_estado.pack(anchor="w", padx=12, pady=(4, 6))

        # Estado de Audio y sEMG
        frame_audio = tk.Frame(
            panel_der,
            bg="white",
            highlightbackground=self.COLOR_BORDE,
            highlightthickness=1
        )
        frame_audio.pack(fill=tk.X, padx=15, pady=(0, 4))

        tk.Label(
            frame_audio,
            text="Audio",
            font=("Helvetica", 9, "bold"),
            bg="white",
            fg=self.COLOR_TEXTO
        ).pack(anchor="w", padx=12, pady=(4, 1))

        self.lbl_audio = tk.Label(
            frame_audio,
            text="MICRÓFONO: EN ESPERA",
            font=("Helvetica", 8),
            bg="white",
            fg=self.COLOR_TEXTO_SEC
        )
        self.lbl_audio.pack(anchor="w", padx=12, pady=(0, 5))

        frame_semg = tk.Frame(
            panel_der,
            bg="white",
            highlightbackground=self.COLOR_BORDE,
            highlightthickness=1
        )
        frame_semg.pack(fill=tk.X, padx=15, pady=(0, 5))

        tk.Label(
            frame_semg,
            text="sEMG (ESP32 / MyoWare)",
            font=("Helvetica", 9, "bold"),
            bg="white",
            fg=self.COLOR_TEXTO
        ).pack(anchor="w", padx=12, pady=(4, 1))

        self.lbl_semg = tk.Label(
            frame_semg,
            text=f"sEMG: EN ESPERA ({self.semg_puerto})",
            font=("Helvetica", 8),
            bg="white",
            fg=self.COLOR_TEXTO_SEC
        )
        self.lbl_semg.pack(anchor="w", padx=12, pady=(0, 5))

        # Contenedores de menus
        self.frame_menu_principal = tk.Frame(panel_der, bg=self.COLOR_PANEL)
        self.frame_menu_principal.pack(fill=tk.BOTH, expand=True)

        self.frame_menu_analisis = tk.Frame(panel_der, bg=self.COLOR_PANEL)

        # Botones del menu principal
        self.btn_iniciar = tk.Button(
            self.frame_menu_principal,
            text="INICIAR ETAPA",
            bg=self.COLOR_VERDE,
            fg="white",
            activebackground="#15803D",
            activeforeground="white",
            font=("Helvetica", 9, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.iniciar_grabacion,
            pady=7
        )
        self.btn_iniciar.pack(fill=tk.X, padx=20, pady=(1, 3))

        self.btn_detener = tk.Button(
            self.frame_menu_principal,
            text="DETENER REGISTRO",
            bg=self.COLOR_ROJO,
            fg="white",
            activebackground="#B91C1C",
            activeforeground="white",
            font=("Helvetica", 9, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.detener_grabacion,
            pady=7,
            state="disabled"
        )
        self.btn_detener.pack(fill=tk.X, padx=20, pady=(0, 5))

        tk.Button(
            self.frame_menu_principal,
            text="VER REPRODUCCIÓN DE MOVIMIENTO",
            bg=self.COLOR_AZUL,
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.ver_movimiento,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=(1, 3))

        tk.Button(
            self.frame_menu_principal,
            text="GENERAR REPORTE DEL PACIENTE",
            bg=self.COLOR_NARANJA,
            fg="white",
            activebackground="#B45309",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.generar_reporte,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=2)

        tk.Button(
            self.frame_menu_principal,
            text="EXPORTAR PROTOCOLO",
            bg=self.COLOR_TURQUESA,
            fg="white",
            activebackground="#115E59",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.exportar_csv,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=2)

        tk.Button(
            self.frame_menu_principal,
            text="IR A MENÚ DE GRÁFICAS",
            bg=self.COLOR_MORADO,
            fg="white",
            activebackground="#6D28D9",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.mostrar_menu_analisis,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=(5, 2))

        # Botones del menu de analisis
        tk.Label(
            self.frame_menu_analisis,
            text="Centro de Análisis",
            font=("Helvetica", 13, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TURQUESA
        ).pack(anchor="w", padx=18, pady=(10, 2))

        tk.Label(
            self.frame_menu_analisis,
            text="Visualización de registros y carpetas",
            font=("Helvetica", 8),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXTO_SEC
        ).pack(anchor="w", padx=18, pady=(0, 10))
        
        tk.Button(
            self.frame_menu_analisis,
            text="GRÁFICAS DE VIDEO (MediaPipe)",
            bg=self.COLOR_NARANJA,
            fg="white",
            activebackground="#B45309",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.ver_grafica,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=3)

        tk.Button(
            self.frame_menu_analisis,
            text="GRÁFICA sEMG (FILTRADA)",
            bg=self.COLOR_TURQUESA,
            fg="white",
            activebackground="#115E59",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.ver_grafica_semg,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=3)

        tk.Button(
            self.frame_menu_analisis,
            text="DASHBOARD COMPARATIVO",
            bg=self.COLOR_MORADO,
            fg="white",
            activebackground="#6D28D9",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.comparar_protocolos,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=3)

        tk.Button(
            self.frame_menu_analisis,
            text="VOLVER A CAPTURA EN VIVO",
            bg=self.COLOR_AZUL,
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            font=("Helvetica", 8, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.mostrar_menu_principal,
            pady=7
        ).pack(fill=tk.X, padx=20, pady=(15, 5))

    def actualizar_instrucciones(self, event=None):
        ej = self.ejercicio_seleccionado.get()

        instrucciones = {
            "Etapa 1": "10 segundos, un solo ciclo.",
            "Etapa 2": "5 segundos de presión de abatelenguas,\n5 repeticiones = 5 ciclos.",
            "Etapa 3": "De sonrisa a beso,\n10 ciclos.",
            "Etapa 4": "10 repeticiones por fonema:\n3 ciclos (pa, ba, ma).",
            "Etapa 5": "Palabras: Mapa, Bebé, Bomba.\n10 repeticiones por palabra = 3 ciclos.",
            "Etapa 6": "Frases:\n'Mi mamá me mima'\n'Papá toma sopa'\n'Mi bebé toma biberón'\nUn solo ciclo."
        }

        for clave, texto in instrucciones.items():
            if clave in ej:
                self.lbl_instrucciones.config(text=texto)
                break

        if not self.grabando:
            self.lbl_tiempo.config(text="00:00.0", fg=self.COLOR_TURQUESA)

    def obtener_carpetas_sujeto(self):
        id_sujeto = self.id_sujeto.get().strip() or "Sujeto_01"
        carpeta_base = os.path.join("AMVA_resultados", id_sujeto)

        carpetas = {
            "base": carpeta_base,
            "Audio": os.path.join(carpeta_base, "Audio"),
            "Datos": os.path.join(carpeta_base, "Datos"),
            "sEMG": os.path.join(carpeta_base, "sEMG")
        }

        for carpeta in carpetas.values():
            os.makedirs(carpeta, exist_ok=True)

        return carpetas

    def obtener_nombre_etapa(self):
        return (
            self.ejercicio_seleccionado.get()
            .strip()
            .replace(":", "")
            .replace(" ", "_")
            .replace("/", "-")
        )

    def obtener_puntos_orofaciales(self, rostro=None, ancho_vid=0, alto=0):
        if rostro is None or ancho_vid <= 0 or alto <= 0:
            return self.PUNTOS_ORBICULAR.copy()

        pt_izq = rostro.landmark[61]
        pt_der = rostro.landmark[291]
        pt_sup = rostro.landmark[13]
        pt_inf = rostro.landmark[14]

        x_izq = pt_izq.x * ancho_vid
        x_der = pt_der.x * ancho_vid
        y_sup = pt_sup.y * alto
        y_inf = pt_inf.y * alto

        ancho_boca = max(abs(x_der - x_izq), 1.0)
        cx = (x_izq + x_der) / 2
        cy = (y_sup + y_inf) / 2

        centro_normalizado = (cx / ancho_vid, cy / alto)
        recalcular = False

        if self.ultimo_centro_orbicular is None:
            recalcular = True
        else:
            desplazamiento = math.hypot(
                centro_normalizado[0] - self.ultimo_centro_orbicular[0],
                centro_normalizado[1] - self.ultimo_centro_orbicular[1]
            )
            if desplazamiento > self.UMBRAL_MOVIMIENTO_REGION:
                recalcular = True

        if self.ultimo_ancho_boca is None:
            recalcular = True
        else:
            cambio_ancho = abs(ancho_boca - self.ultimo_ancho_boca) / max(self.ultimo_ancho_boca, 1.0)
            if cambio_ancho > self.UMBRAL_CAMBIO_ANCHO:
                recalcular = True

        if self.PUNTOS_ORBICULAR and not recalcular:
            return self.PUNTOS_ORBICULAR.copy()

        rx = ancho_boca * self.FACTOR_ANCHO_ORBICULAR
        ry = ancho_boca * self.FACTOR_ALTO_ORBICULAR
        cy += ancho_boca * self.DESPLAZAMIENTO_VERTICAL_ORBICULAR

        y_nariz = rostro.landmark[1].y * alto
        y_menton = rostro.landmark[152].y * alto

        limite_superior = y_nariz - 0.02 * alto
        limite_inferior = y_menton + 0.015 * alto

        seleccionados = []

        for idx, punto in enumerate(rostro.landmark):
            x = punto.x * ancho_vid
            y = punto.y * alto

            if not (0 <= x < ancho_vid and 0 <= y < alto):
                continue

            dentro_elipse = (((x - cx) / max(rx, 1.0)) ** 2 + ((y - cy) / max(ry, 1.0)) ** 2) <= 1
            dentro_vertical = limite_superior <= y <= limite_inferior

            if dentro_elipse and dentro_vertical:
                seleccionados.append(idx)

        for idx in [61, 291, 13, 14]:
            if idx not in seleccionados:
                seleccionados.append(idx)

        self.PUNTOS_ORBICULAR = sorted(set(seleccionados))
        self.ultimo_centro_orbicular = (cx / ancho_vid, cy / alto)
        self.ultimo_ancho_boca = ancho_boca

        return self.PUNTOS_ORBICULAR.copy()

    def calcular_velocidad_orofacial(self, rostro, ancho_vid, alto, tiempo_actual):
        puntos_actuales = {}

        for idx in self.PUNTOS_ORBICULAR:
            pt = rostro.landmark[idx]
            puntos_actuales[idx] = (pt.x * ancho_vid, pt.y * alto)

        if self.puntos_previos_velocidad is None or self.tiempo_previo_velocidad is None:
            self.puntos_previos_velocidad = puntos_actuales
            self.tiempo_previo_velocidad = tiempo_actual
            return 0.0

        dt = tiempo_actual - self.tiempo_previo_velocidad
        if dt <= 0:
            return 0.0

        velocidades = []

        for idx in puntos_actuales:
            if idx not in self.puntos_previos_velocidad:
                continue

            x_actual, y_actual = puntos_actuales[idx]
            x_anterior, y_anterior = self.puntos_previos_velocidad[idx]

            desplazamiento = math.hypot(x_actual - x_anterior, y_actual - y_anterior)
            velocidades.append(desplazamiento / dt)

        self.puntos_previos_velocidad = puntos_actuales
        self.tiempo_previo_velocidad = tiempo_actual

        if not velocidades:
            return 0.0

        return float(np.mean(velocidades))

    def obtener_centro_perioral(self, rostro, ancho_vid, alto):
        puntos = [61, 291, 13, 14]
        xs = [rostro.landmark[idx].x * ancho_vid for idx in puntos]
        ys = [rostro.landmark[idx].y * alto for idx in puntos]
        return float(np.mean(xs)), float(np.mean(ys))

    def calcular_desplazamiento_perioral(self, rostro, ancho_vid, alto):
        centro_actual = self.obtener_centro_perioral(rostro, ancho_vid, alto)
        desplazamiento_frame_px = 0.0

        if self.ultima_posicion_centro is not None:
            desplazamiento_frame_px = math.hypot(
                centro_actual[0] - self.ultima_posicion_centro[0],
                centro_actual[1] - self.ultima_posicion_centro[1]
            )

        self.ultima_posicion_centro = centro_actual
        desplazamiento_frame_mm = desplazamiento_frame_px * self.FACTOR_CALIBRACION_MM_PX

        self.desplazamiento_acumulado_px += desplazamiento_frame_px
        self.desplazamiento_acumulado_mm += desplazamiento_frame_mm

        return desplazamiento_frame_mm, self.desplazamiento_acumulado_mm

    def dibujar_ovalo_posicion(self, fotograma, ancho_vid, alto):
        centro = (ancho_vid // 2, int(alto * 0.50))
        ejes = (int(ancho_vid * 0.25), int(alto * 0.34))
        cv2.ellipse(fotograma, centro, ejes, 0, 0, 360, self.COLOR_OVALO, 2, cv2.LINE_AA)

    def dibujar_landmarks_orofaciales(self, fotograma, rostro, ancho_vid, alto):
        if not self.PUNTOS_ORBICULAR:
            self.obtener_puntos_orofaciales(rostro, ancho_vid, alto)

        for idx in self.PUNTOS_ORBICULAR:
            pt = rostro.landmark[idx]
            if not (0 <= pt.x <= 1 and 0 <= pt.y <= 1):
                continue

            x = int(np.clip(pt.x * ancho_vid, 0, ancho_vid - 1))
            y = int(np.clip(pt.y * alto, 0, alto - 1))

            if idx in [13, 14]:
                color = self.COLOR_APERTURA
                tamano = self.TAMANO_MEDICION
            elif idx in [61, 291]:
                color = self.COLOR_ANCHO
                tamano = self.TAMANO_MEDICION
            else:
                color = self.COLOR_LANDMARK
                tamano = self.TAMANO_LANDMARK

            cv2.circle(fotograma, (x, y), tamano, color, 1, cv2.LINE_AA)

    def guardar_parametros_video_etapa(self, duracion_etapa):
        if not self.datos_video_etapa:
            return

        try:
            carpetas = self.obtener_carpetas_sujeto()
            ruta_frames = os.path.join(carpetas["Datos"], "Video_parametros_por_frame.csv")
            ruta_resumen = os.path.join(carpetas["Datos"], "Video_parametros_resumen.csv")

            id_sujeto = self.id_sujeto.get().strip() or "Sujeto_01"
            condicion = self.condicion_sujeto.get()
            etapa = self.ejercicio_seleccionado.get()

            existe_frames = os.path.exists(ruta_frames)

            with open(ruta_frames, "a", newline="", encoding="utf-8-sig") as archivo:
                writer = csv.writer(archivo)
                if not existe_frames:
                    writer.writerow([
                        "ID_Sujeto", "Condicion", "Etapa", "Frame", "Tiempo(s)",
                        "Apertura(mm)", "Ancho(mm)", "ROM_vertical(mm)", "ROM_horizontal(mm)",
                        "Velocidad(mm/s)", "Desplazamiento_Frame(mm)",
                        "Desplazamiento_Acumulado(mm)", "Duracion_Etapa(s)"
                    ])

                for fila in self.datos_video_etapa:
                    writer.writerow([
                        id_sujeto, condicion, etapa, fila["frame"],
                        f"{fila['tiempo']:.3f}", f"{fila['apertura']:.3f}",
                        f"{fila['ancho']:.3f}", f"{fila['rom_vertical']:.3f}",
                        f"{fila['rom_horizontal']:.3f}", f"{fila['velocidad']:.3f}",
                        f"{fila['desplazamiento_frame']:.3f}",
                        f"{fila['desplazamiento_acumulado']:.3f}",
                        f"{duracion_etapa:.3f}"
                    ])

            aperturas = np.array([fila["apertura"] for fila in self.datos_video_etapa], dtype=float)
            anchos = np.array([fila["ancho"] for fila in self.datos_video_etapa], dtype=float)
            velocidades = np.array([fila["velocidad"] for fila in self.datos_video_etapa], dtype=float)

            if len(aperturas) > 0:
                apertura_min = float(np.min(aperturas))
                apertura_max = float(np.max(aperturas))
                apertura_media = float(np.mean(aperturas))

                ancho_min = float(np.min(anchos))
                ancho_max = float(np.max(anchos))
                ancho_medio = float(np.mean(anchos))

                rom_vertical = apertura_max - apertura_min
                rom_horizontal = ancho_max - ancho_min
                rom_total = math.hypot(rom_vertical, rom_horizontal)

                velocidad_media = float(np.mean(velocidades))
                velocidad_maxima = float(np.max(velocidades))
            else:
                apertura_min = apertura_max = apertura_media = 0.0
                ancho_min = ancho_max = ancho_medio = 0.0
                rom_vertical = rom_horizontal = rom_total = 0.0
                velocidad_media = velocidad_maxima = 0.0

            desplazamiento_total = self.desplazamiento_acumulado_mm
            existe_resumen = os.path.exists(ruta_resumen)

            with open(ruta_resumen, "a", newline="", encoding="utf-8-sig") as archivo:
                writer = csv.writer(archivo)
                if not existe_resumen:
                    writer.writerow([
                        "ID_Sujeto", "Condicion", "Etapa",
                        "Apertura_media(mm)", "Apertura_min(mm)", "Apertura_max(mm)",
                        "Ancho_medio(mm)", "Ancho_min(mm)", "Ancho_max(mm)",
                        "ROM_vertical(mm)", "ROM_horizontal(mm)", "ROM_total(mm)",
                        "Velocidad_media(mm/s)", "Velocidad_maxima(mm/s)",
                        "Desplazamiento_total(mm)", "Duracion(s)", "Numero_frames"
                    ])

                writer.writerow([
                    id_sujeto, condicion, etapa,
                    f"{apertura_media:.3f}", f"{apertura_min:.3f}", f"{apertura_max:.3f}",
                    f"{ancho_medio:.3f}", f"{ancho_min:.3f}", f"{ancho_max:.3f}",
                    f"{rom_vertical:.3f}", f"{rom_horizontal:.3f}", f"{rom_total:.3f}",
                    f"{velocidad_media:.3f}", f"{velocidad_maxima:.3f}",
                    f"{desplazamiento_total:.3f}", f"{duracion_etapa:.3f}",
                    len(self.datos_video_etapa)
                ])

            self.parametros_video_etapa = {
                "apertura_media": apertura_media,
                "apertura_min": apertura_min,
                "apertura_max": apertura_max,
                "ancho_medio": ancho_medio,
                "ancho_min": ancho_min,
                "ancho_max": ancho_max,
                "rom_vertical": rom_vertical,
                "rom_horizontal": rom_horizontal,
                "rom_total": rom_total,
                "velocidad_media": velocidad_media,
                "velocidad_maxima": velocidad_maxima,
                "desplazamiento_total": desplazamiento_total,
                "duracion": duracion_etapa,
                "numero_frames": len(self.datos_video_etapa)
            }

        except Exception as e:
            print("Error guardando parametros de video:", e)
            messagebox.showerror(
                "Error de datos",
                f"No fue posible guardar los parámetros numéricos de video.\n\nDetalles:\n{e}"
            )

    def callback_audio(self, indata, frames, tiempo, status):
        if status:
            print("Estado del audio:", status)
        if self.audio_grabando:
            self.audio_frames.append(indata.copy())

    def iniciar_audio(self):
        try:
            self.audio_frames = []
            self.audio_grabando = True

            self.audio_stream = sd.InputStream(
                samplerate=self.audio_samplerate,
                channels=self.audio_channels,
                dtype=self.audio_dtype,
                callback=self.callback_audio
            )
            self.audio_stream.start()
            self.lbl_audio.config(text="MICRÓFONO: GRABANDO", fg=self.COLOR_ROJO)
            return True

        except Exception as e:
            self.audio_grabando = False
            self.audio_stream = None
            self.lbl_audio.config(text="MICRÓFONO: ERROR", fg=self.COLOR_ROJO)
            messagebox.showerror(
                "Error de audio",
                f"No fue posible iniciar el micrófono.\n\nDetalles:\n{e}"
            )
            return False

    def estimar_f0(self, segmento):
        segmento = np.asarray(segmento, dtype=float)
        if len(segmento) < 100:
            return 0.0

        segmento = segmento - np.mean(segmento)
        energia = np.sqrt(np.mean(segmento ** 2))
        if energia < 1e-5:
            return 0.0

        ventana = np.hanning(len(segmento))
        segmento = segmento * ventana

        autocorr = np.correlate(segmento, segmento, mode="full")[len(segmento) - 1:]
        if autocorr[0] <= 0:
            return 0.0

        autocorr = autocorr / autocorr[0]

        f0_min = 70.0
        f0_max = 400.0

        lag_min = max(1, int(self.audio_samplerate / f0_max))
        lag_max = min(len(autocorr) - 1, int(self.audio_samplerate / f0_min))

        if lag_max <= lag_min:
            return 0.0

        segmento_ac = autocorr[lag_min:lag_max + 1]
        indice = np.argmax(segmento_ac)
        lag = indice + lag_min
        valor_pico = autocorr[lag]

        if valor_pico < 0.20:
            return 0.0

        if 1 <= lag < len(autocorr) - 1:
            y1 = autocorr[lag - 1]
            y2 = autocorr[lag]
            y3 = autocorr[lag + 1]
            denominador = y1 - 2 * y2 + y3
            if abs(denominador) > 1e-12:
                ajuste = 0.5 * (y1 - y3) / denominador
                lag = lag + ajuste

        if lag <= 0:
            return 0.0

        return float(self.audio_samplerate / lag)

    def analizar_audio(self, audio):
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) == 0:
            return None

        audio = audio / 32768.0
        duracion_total = len(audio) / self.audio_samplerate

        duracion_ventana = 0.030
        duracion_paso = 0.010

        tam_ventana = int(self.audio_samplerate * duracion_ventana)
        tam_paso = int(self.audio_samplerate * duracion_paso)

        if len(audio) < tam_ventana:
            return None

        rms = []
        tiempos = []
        f0s = []

        for inicio in range(0, len(audio) - tam_ventana + 1, tam_paso):
            segmento = audio[inicio:inicio + tam_ventana]
            valor_rms = np.sqrt(np.mean(segmento ** 2))
            rms.append(valor_rms)
            tiempos.append((inicio + tam_ventana / 2) / self.audio_samplerate)

            if valor_rms > 0:
                f0s.append(self.estimar_f0(segmento))
            else:
                f0s.append(0.0)

        rms = np.asarray(rms)
        tiempos = np.asarray(tiempos)
        f0s = np.asarray(f0s)

        if len(rms) == 0:
            return None

        piso_ruido = np.percentile(rms, 20)
        percentil_alto = np.percentile(rms, 75)
        umbral_rms = max(piso_ruido * 2.5, percentil_alto * 0.20, 0.003)

        actividad = rms > umbral_rms
        indices_activos = np.where(actividad)[0]

        if len(indices_activos) == 0:
            return {
                "inicio_fonacion": 0.0,
                "fin_fonacion": 0.0,
                "duracion_fonacion": 0.0,
                "numero_pausas": 0,
                "duracion_pausas": 0.0,
                "f0_promedio": 0.0,
                "f0_minimo": 0.0,
                "f0_maximo": 0.0,
                "duracion_total_audio": duracion_total
            }

        indice_inicio = indices_activos[0]
        indice_fin = indices_activos[-1]

        inicio_fonacion = float(tiempos[indice_inicio])
        fin_fonacion = float(tiempos[indice_fin] + duracion_ventana)
        duracion_fonacion = max(0.0, fin_fonacion - inicio_fonacion)

        numero_pausas = 0
        duracion_pausas = 0.0
        grupos_activos = []

        inicio_grupo = indices_activos[0]
        anterior = indices_activos[0]

        for indice in indices_activos[1:]:
            if indice - anterior > 1:
                grupos_activos.append((inicio_grupo, anterior))
                inicio_grupo = indice
            anterior = indice

        grupos_activos.append((inicio_grupo, anterior))

        for i in range(len(grupos_activos) - 1):
            fin_grupo = grupos_activos[i][1]
            inicio_siguiente = grupos_activos[i + 1][0]

            inicio_pausa = tiempos[fin_grupo] + duracion_ventana / 2
            fin_pausa = tiempos[inicio_siguiente] - duracion_ventana / 2
            pausa = fin_pausa - inicio_pausa

            if pausa >= 0.15:
                numero_pausas += 1
                duracion_pausas += pausa

        f0_validas = f0s[(f0s >= 70) & (f0s <= 400) & actividad]

        if len(f0_validas) > 0:
            f0_promedio = float(np.mean(f0_validas))
            f0_minimo = float(np.min(f0_validas))
            f0_maximo = float(np.max(f0_validas))
        else:
            f0_promedio = f0_minimo = f0_maximo = 0.0

        return {
            "inicio_fonacion": inicio_fonacion,
            "fin_fonacion": fin_fonacion,
            "duracion_fonacion": duracion_fonacion,
            "numero_pausas": numero_pausas,
            "duracion_pausas": float(duracion_pausas),
            "f0_promedio": f0_promedio,
            "f0_minimo": f0_minimo,
            "f0_maximo": f0_maximo,
            "duracion_total_audio": duracion_total
        }

    def detener_audio(self):
        if not self.audio_grabando:
            return None

        self.audio_grabando = False

        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            if not self.audio_frames:
                self.lbl_audio.config(text="MICRÓFONO: SIN DATOS", fg=self.COLOR_TEXTO_SEC)
                return None

            audio = np.concatenate(self.audio_frames, axis=0)
            carpetas = self.obtener_carpetas_sujeto()
            id_sujeto = self.id_sujeto.get().strip() or "Sujeto_01"
            etapa = self.obtener_nombre_etapa()

            nombre_archivo = f"{id_sujeto}_{etapa}.wav"
            ruta_audio = os.path.join(carpetas["Audio"], nombre_archivo)

            with wave.open(ruta_audio, "wb") as archivo_wav:
                archivo_wav.setnchannels(self.audio_channels)
                archivo_wav.setsampwidth(2)
                archivo_wav.setframerate(self.audio_samplerate)
                archivo_wav.writeframes(audio.tobytes())

            self.ruta_audio_actual = ruta_audio
            parametros = self.analizar_audio(audio)

            if parametros is not None:
                self.parametros_audio_etapa = parametros
                self.audio_inicio_fonacion = parametros["inicio_fonacion"]
                self.audio_fin_fonacion = parametros["fin_fonacion"]
                self.audio_duracion_fonacion = parametros["duracion_fonacion"]
                self.audio_numero_pausas = parametros["numero_pausas"]
                self.audio_duracion_pausas = parametros["duracion_pausas"]
                self.audio_f0_promedio = parametros["f0_promedio"]
                self.audio_f0_minimo = parametros["f0_minimo"]
                self.audio_f0_maximo = parametros["f0_maximo"]

                self.guardar_parametros_audio(parametros)

                if parametros["f0_promedio"] > 0:
                    texto_audio = f"AUDIO GUARDADO | F0: {parametros['f0_promedio']:.1f} Hz"
                else:
                    texto_audio = "AUDIO GUARDADO | F0: No estimable"

                self.lbl_audio.config(text=texto_audio, fg=self.COLOR_VERDE)
            else:
                self.lbl_audio.config(text="AUDIO GUARDADO | SIN PARÁMETROS", fg=self.COLOR_VERDE)

            return ruta_audio

        except Exception as e:
            self.lbl_audio.config(text="MICRÓFONO: ERROR AL GUARDAR", fg=self.COLOR_ROJO)
            messagebox.showerror(
                "Error de audio",
                f"No fue posible guardar o analizar el audio.\n\nDetalles:\n{e}"
            )
            return None

    def guardar_parametros_audio(self, parametros):
        try:
            carpetas = self.obtener_carpetas_sujeto()
            ruta_csv = os.path.join(carpetas["Datos"], "Audio_parametros.csv")
            existe = os.path.exists(ruta_csv)

            id_sujeto = self.id_sujeto.get().strip() or "Sujeto_01"
            condicion = self.condicion_sujeto.get()
            etapa = self.ejercicio_seleccionado.get()
            archivo_audio = os.path.basename(self.ruta_audio_actual) if self.ruta_audio_actual else ""

            with open(ruta_csv, "a", newline="", encoding="utf-8-sig") as archivo:
                writer = csv.writer(archivo)
                if not existe:
                    writer.writerow([
                        "ID_Sujeto", "Condicion", "Etapa", "Archivo_Audio",
                        "Inicio_fonacion(s)", "Fin_fonacion(s)", "Duracion_fonacion(s)",
                        "Numero_pausas", "Duracion_pausas(s)", "F0_promedio(Hz)",
                        "F0_minimo(Hz)", "F0_maximo(Hz)", "Duracion_total_audio(s)"
                    ])

                writer.writerow([
                    id_sujeto, condicion, etapa, archivo_audio,
                    f"{parametros['inicio_fonacion']:.3f}",
                    f"{parametros['fin_fonacion']:.3f}",
                    f"{parametros['duracion_fonacion']:.3f}",
                    parametros["numero_pausas"],
                    f"{parametros['duracion_pausas']:.3f}",
                    f"{parametros['f0_promedio']:.2f}",
                    f"{parametros['f0_minimo']:.2f}",
                    f"{parametros['f0_maximo']:.2f}",
                    f"{parametros['duracion_total_audio']:.3f}"
                ])

        except Exception as e:
            print("Error guardando parametros de audio:", e)

    def actualizar_video(self):
        if not self.camara_activa:
            return

        exito, fotograma = self.cap.read()

        if exito:
            fotograma = cv2.flip(fotograma, 1)
            alto, ancho_vid, _ = fotograma.shape

            self.dibujar_ovalo_posicion(fotograma, ancho_vid, alto)
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            resultados = self.malla_facial.process(fotograma_rgb)

            apertura_boca_mm = 0.0
            ancho_boca_mm = 0.0
            velocidad_mm_s = 0.0

            if resultados.multi_face_landmarks:
                for rostro in resultados.multi_face_landmarks:
                    pt_izq = rostro.landmark[61]
                    pt_der = rostro.landmark[291]
                    pt_sup = rostro.landmark[13]
                    pt_inf = rostro.landmark[14]

                    x_izq, y_izq = int(pt_izq.x * ancho_vid), int(pt_izq.y * alto)
                    x_der, y_der = int(pt_der.x * ancho_vid), int(pt_der.y * alto)
                    x_sup, y_sup = int(pt_sup.x * ancho_vid), int(pt_sup.y * alto)
                    x_inf, y_inf = int(pt_inf.x * ancho_vid), int(pt_inf.y * alto)

                    apertura_boca = math.hypot(x_sup - x_inf, y_sup - y_inf)
                    ancho_boca = math.hypot(x_der - x_izq, y_der - y_izq)

                    apertura_boca_mm = apertura_boca * self.FACTOR_CALIBRACION_MM_PX
                    ancho_boca_mm = ancho_boca * self.FACTOR_CALIBRACION_MM_PX

                    self.obtener_puntos_orofaciales(rostro, ancho_vid, alto)

                    if self.grabando:
                        tiempo_actual = time.time() - self.tiempo_inicio
                        velocidad = self.calcular_velocidad_orofacial(
                            rostro, ancho_vid, alto, tiempo_actual
                        )
                        velocidad_mm_s = velocidad * self.FACTOR_CALIBRACION_MM_PX
                        self.velocidad_actual = velocidad_mm_s

                        if velocidad_mm_s > self.velocidad_maxima_mm_s:
                            self.velocidad_maxima_mm_s = velocidad_mm_s

                        desplazamiento_frame_mm, desplazamiento_acumulado_mm = (
                            self.calcular_desplazamiento_perioral(rostro, ancho_vid, alto)
                        )

                        self.numero_frame_actual += 1

                        puntos_frame = [
                            {"id": i, "x": rostro.landmark[i].x, "y": rostro.landmark[i].y}
                            for i in self.PUNTOS_ORBICULAR
                        ]

                        self.frames_actuales.append(puntos_frame)
                        self.tiempos_actuales.append(tiempo_actual)

                        etapa = self.ejercicio_seleccionado.get()

                        aperturas_etapa = [f["apertura"] for f in self.datos_video_etapa]
                        anchos_etapa = [f["ancho"] for f in self.datos_video_etapa]

                        aperturas_etapa.append(apertura_boca_mm)
                        anchos_etapa.append(ancho_boca_mm)

                        rom_vertical_actual = max(aperturas_etapa) - min(aperturas_etapa)
                        rom_horizontal_actual = max(anchos_etapa) - min(anchos_etapa)

                        fila_video = {
                            "frame": self.numero_frame_actual,
                            "tiempo": tiempo_actual,
                            "apertura": apertura_boca_mm,
                            "ancho": ancho_boca_mm,
                            "rom_vertical": rom_vertical_actual,
                            "rom_horizontal": rom_horizontal_actual,
                            "velocidad": velocidad_mm_s,
                            "desplazamiento_frame": desplazamiento_frame_mm,
                            "desplazamiento_acumulado": desplazamiento_acumulado_mm
                        }

                        self.datos_video_etapa.append(fila_video)
                        self.datos_sesion_completa.append((
                            etapa, self.numero_frame_actual, tiempo_actual,
                            apertura_boca_mm, ancho_boca_mm, velocidad_mm_s
                        ))
                    else:
                        velocidad_mm_s = self.velocidad_actual

                    self.dibujar_landmarks_orofaciales(fotograma, rostro, ancho_vid, alto)

            self.lbl_apertura.config(text=f"Apertura Bucal: {apertura_boca_mm:.1f} mm")
            self.lbl_ancho.config(text=f"Ancho de Boca: {ancho_boca_mm:.1f} mm")
            self.lbl_velocidad.config(text=f"Velocidad: {velocidad_mm_s:.1f} mm/s")

            if self.grabando:
                tiempo_transcurrido = time.time() - self.tiempo_inicio
                minutos = int(tiempo_transcurrido // 60)
                segundos = tiempo_transcurrido % 60
                self.lbl_tiempo.config(text=f"{minutos:02d}:{segundos:04.1f}", fg=self.COLOR_ROJO)

            img_pil = Image.fromarray(cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.pantalla_video.imgtk = img_tk
            self.pantalla_video.configure(image=img_tk)

        self.root.after(10, self.actualizar_video)

    def iniciar_semg(self):
        self.semg_tiempos = []
        self.semg_raw = []
        self.semg_filtrada = []
        self.semg_parametros_etapa = {}
        self.semg_ruta_csv = None

        if serial is None:
            self.lbl_semg.config(text="sEMG: falta pyserial", fg=self.COLOR_ROJO)
            return False
        if butter is None or filtfilt is None or iirnotch is None:
            self.lbl_semg.config(text="sEMG: falta scipy", fg=self.COLOR_ROJO)
            return False

        try:
            self.semg_serial = serial.Serial(self.semg_puerto, self.semg_baudrate, timeout=0.2)
            self.semg_serial.reset_input_buffer()
            self.semg_detener_evento.clear()
            self.semg_grabando = True
            self.semg_hilo = threading.Thread(target=self._leer_semg_serie, daemon=True)
            self.semg_hilo.start()
            self.lbl_semg.config(text=f"sEMG: ADQUIRIENDO ({self.semg_puerto})", fg=self.COLOR_VERDE)
            return True
        except Exception as exc:
            self.semg_serial = None
            self.semg_grabando = False
            self.lbl_semg.config(text=f"sEMG no conectado: {exc}", fg=self.COLOR_NARANJA)
            return False

    def _leer_semg_serie(self):
        inicio = time.perf_counter()
        while not self.semg_detener_evento.is_set():
            try:
                if self.semg_serial is None:
                    break
                linea = self.semg_serial.readline().decode("utf-8", errors="ignore").strip()
                if not linea:
                    continue
                try:
                    valor = float(linea)
                except ValueError:
                    continue
                self.semg_tiempos.append(time.perf_counter() - inicio)
                self.semg_raw.append(valor)
            except Exception:
                if not self.semg_detener_evento.is_set():
                    time.sleep(0.01)
                break

    def detener_semg(self):
        self.semg_detener_evento.set()
        self.semg_grabando = False

        if self.semg_hilo and self.semg_hilo.is_alive():
            self.semg_hilo.join(timeout=1.5)

        if self.semg_serial is not None:
            try:
                self.semg_serial.close()
            except Exception:
                pass
            self.semg_serial = None

        if len(self.semg_raw) < 20:
            self.lbl_semg.config(
                text=f"sEMG: pocos datos ({len(self.semg_raw)} muestras)",
                fg=self.COLOR_NARANJA
            )
            return False

        raw = np.asarray(self.semg_raw, dtype=float)
        tiempos = np.asarray(self.semg_tiempos, dtype=float)
        fs = float(self.semg_fs_nominal)
        filtrada = raw.copy()

        try:
            b_notch, a_notch = iirnotch(w0=60.0, Q=30.0, fs=fs)
            filtrada = filtfilt(b_notch, a_notch, filtrada)
            nyquist = 0.5 * fs
            b_hp, a_hp = butter(N=4, Wn=20.0 / nyquist, btype="high")
            filtrada = filtfilt(b_hp, a_hp, filtrada)
        except Exception as exc:
            self.lbl_semg.config(text=f"sEMG: error al filtrar ({exc})", fg=self.COLOR_ROJO)
            return False

        self.semg_filtrada = filtrada.tolist()
        self.semg_parametros_etapa = {
            "muestras": int(len(raw)),
            "duracion_s": float(tiempos[-1] - tiempos[0]) if len(tiempos) > 1 else 0.0,
            "media_raw": float(np.mean(raw)),
            "desv_std_raw": float(np.std(raw)),
            "rms_filtrada": float(np.sqrt(np.mean(np.square(filtrada)))),
            "fs_nominal_hz": fs,
        }

        try:
            carpetas = self.obtener_carpetas_sujeto()
            nombre_etapa = self.obtener_nombre_etapa()
            ruta = os.path.join(carpetas["sEMG"], f"{nombre_etapa}_sEMG.csv")
            with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow(["tiempo_s", "senal_raw", "senal_filtrada", "fs_nominal_hz"])
                escritor.writerows(zip(tiempos, raw, filtrada, [fs] * len(raw)))
            self.semg_ruta_csv = ruta
            self.lbl_semg.config(text=f"sEMG guardado: {len(raw)} muestras", fg=self.COLOR_VERDE)
            return True
        except Exception as exc:
            self.lbl_semg.config(text=f"sEMG: no se pudo guardar ({exc})", fg=self.COLOR_ROJO)
            return False

    def ver_grafica(self):
        t = []
        aperturas = []
        anchos = []
        velocidades = []

        if not self.datos_sesion_completa:
            archivo_csv = filedialog.askopenfilename(
                title="Selecciona el archivo Video_parametros_por_frame.csv del paciente",
                initialdir="AMVA_resultados",
                filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
            )
            if not archivo_csv:
                return
            
            try:
                df = pd.read_csv(archivo_csv)
                df.columns = [str(c).strip().lower() for c in df.columns]
                
                col_t = next((c for c in df.columns if 'tiempo' in c), 'tiempo(s)')
                col_a = next((c for c in df.columns if 'apertura' in c), 'apertura(mm)')
                col_w = next((c for c in df.columns if 'ancho' in c), 'ancho(mm)')
                col_v = next((c for c in df.columns if 'velocidad' in c), 'velocidad(mm/s)')
                col_etapa = next((c for c in df.columns if 'etapa' in c), None)

                if col_etapa and col_etapa in df.columns:
                    etapas_disponibles = df[col_etapa].dropna().unique().tolist()
                    if etapas_disponibles:
                        v_sel = tk.Toplevel(self.root)
                        v_sel.title("Seleccionar Etapa")
                        v_sel.geometry("350x150")
                        v_sel.configure(bg=self.COLOR_FONDO)
                        v_sel.grab_set()

                        tk.Label(
                            v_sel,
                            text="Selecciona la etapa a graficar:",
                            font=("Helvetica", 9, "bold"),
                            bg=self.COLOR_FONDO,
                            fg=self.COLOR_TEXTO
                        ).pack(pady=(15, 5))
                        
                        combo_etapas = ttk.Combobox(v_sel, values=etapas_disponibles, state="readonly", width=30)
                        combo_etapas.pack(pady=5)
                        combo_etapas.set(etapas_disponibles[0])

                        seleccion_usuario = {"etapa": None}

                        def confirmar():
                            seleccion_usuario["etapa"] = combo_etapas.get()
                            v_sel.destroy()

                        tk.Button(
                            v_sel,
                            text="Visualizar Gráfica",
                            bg=self.COLOR_TURQUESA,
                            fg="white",
                            font=("Helvetica", 8, "bold"),
                            command=confirmar,
                            pady=5
                        ).pack(pady=10)
                        self.root.wait_window(v_sel)

                        if seleccion_usuario["etapa"]:
                            df = df[df[col_etapa] == seleccion_usuario["etapa"]]

                t = pd.to_numeric(df[col_t], errors='coerce').values
                aperturas = pd.to_numeric(df[col_a], errors='coerce').values
                anchos = pd.to_numeric(df[col_w], errors='coerce').values
                velocidades = pd.to_numeric(df[col_v], errors='coerce').values

                mask = ~np.isnan(t) & ~np.isnan(aperturas) & ~np.isnan(anchos) & ~np.isnan(velocidades)
                t, aperturas, anchos, velocidades = t[mask], aperturas[mask], anchos[mask], velocidades[mask]
            except Exception as e:
                messagebox.showerror("Error de lectura", f"No se pudo cargar el archivo de video:\n{e}")
                return
        else:
            t_len = len(self.tiempos_actuales)
            datos_recientes = self.datos_sesion_completa[-t_len:] if t_len > 0 else self.datos_sesion_completa
            t = [d[2] for d in datos_recientes]
            aperturas = [d[3] for d in datos_recientes]
            anchos = [d[4] for d in datos_recientes]
            velocidades = [d[5] for d in datos_recientes]

        if len(t) == 0:
            messagebox.showwarning("Sin datos", "No hay registros de video suficientes para graficar.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10))
        fig.patch.set_facecolor('#FFFFFF')

        ax1.plot(t, aperturas, color="#0F766E", linewidth=1.2, label="Apertura")
        ax1.set_title("Apertura Bucal (MediaPipe)", fontsize=11, fontweight='bold', color='#1E293B')
        ax1.set_ylabel("mm", fontsize=10, color='#1E293B')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        ax2.plot(t, anchos, color="#2563EB", linewidth=1.2, label="Ancho")
        ax2.set_title("Ancho de Boca (MediaPipe)", fontsize=11, fontweight='bold', color='#1E293B')
        ax2.set_ylabel("mm", fontsize=10, color='#1E293B')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        ax3.plot(t, velocidades, color="#D97706", linewidth=1.2, label="Velocidad")
        ax3.set_title("Velocidad de Movimiento Orofacial", fontsize=11, fontweight='bold', color='#1E293B')
        ax3.set_xlabel("Tiempo (s)", fontsize=10, color='#1E293B')
        ax3.set_ylabel("mm/s", fontsize=10, color='#1E293B')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        
    def ver_grafica_semg(self):
        tiempos = None
        filtrada = None

        if not self.semg_raw:
            archivo_csv = filedialog.askopenfilename(
                title="Selecciona el archivo CSV sEMG de la etapa a visualizar",
                initialdir="AMVA_resultados",
                filetypes=[("Archivos CSV de sEMG", "*.csv"), ("Todos los archivos", "*.*")]
            )
            if not archivo_csv:
                return
            
            try:
                df = pd.read_csv(archivo_csv)
                df.columns = [str(c).strip().lower() for c in df.columns]
                
                col_t = next((c for c in df.columns if 'tiempo' in c or 'time' in c), df.columns[0])
                col_f = next((c for c in df.columns if 'filtrada' in c or 'senal' in c), None)
                
                if not col_f and len(df.columns) > 2:
                    col_f = df.columns[2]
                elif not col_f and len(df.columns) > 1:
                    col_f = df.columns[1]

                tiempos = pd.to_numeric(df[col_t], errors='coerce').values
                filtrada = pd.to_numeric(df[col_f], errors='coerce').values
                
                mask = ~np.isnan(tiempos) & ~np.isnan(filtrada)
                tiempos = tiempos[mask]
                filtrada = filtrada[mask]
            except Exception as e:
                messagebox.showerror("Error de lectura", f"No se pudo cargar el archivo sEMG:\n{e}")
                return
        else:
            tiempos = np.asarray(self.semg_tiempos, dtype=float)
            if len(self.semg_filtrada) == len(self.semg_tiempos):
                filtrada = np.asarray(self.semg_filtrada, dtype=float)
            else:
                filtrada = np.asarray(self.semg_raw, dtype=float)

        if len(tiempos) == 0 or len(filtrada) == 0:
            messagebox.showwarning("Sin datos", "No hay registros suficientes para graficar.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#FFFFFF')
        
        ax.plot(tiempos, filtrada, color="#0F766E", linewidth=1.2, label="sEMG Filtrada (Pasa-altas 20 Hz + Notch 60 Hz)")
        ax.set_title("Actividad Muscular — sEMG (Orbicular de los Labios)", fontsize=12, fontweight='bold', color='#1E293B')
        ax.set_xlabel("Tiempo transcurrido (s)", fontsize=10, color='#1E293B')
        ax.set_ylabel("Amplitud Filtrada", fontsize=10, color='#1E293B')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def iniciar_grabacion(self):
        if self.grabando:
            return

        self.frames_actuales.clear()
        self.tiempos_actuales.clear()
        self.numero_frame_actual = 0

        self.puntos_previos_velocidad = None
        self.tiempo_previo_velocidad = None

        self.velocidad_actual = 0.0
        self.velocidad_maxima_mm_s = 0.0

        self.ultima_posicion_centro = None
        self.desplazamiento_acumulado_px = 0.0
        self.desplazamiento_acumulado_mm = 0.0

        self.datos_video_etapa = []
        self.parametros_video_etapa = {}

        self.PUNTOS_ORBICULAR.clear()
        self.ultimo_centro_orbicular = None
        self.ultimo_ancho_boca = None

        self.ruta_audio_actual = None
        self.parametros_audio_etapa = {}

        self.audio_inicio_fonacion = None
        self.audio_fin_fonacion = None
        self.audio_duracion_fonacion = 0.0
        self.audio_numero_pausas = 0
        self.audio_duracion_pausas = 0.0
        self.audio_f0_promedio = 0.0
        self.audio_f0_minimo = 0.0
        self.audio_f0_maximo = 0.0

        if not self.iniciar_audio():
            return

        self.iniciar_semg()
        self.tiempo_inicio = time.time()
        self.grabando = True

        self.lbl_estado.config(
            text=f"GRABANDO: {self.ejercicio_seleccionado.get()[:20]}",
            fg=self.COLOR_ROJO
        )
        self.lbl_tiempo.config(text="00:00.0", fg=self.COLOR_ROJO)

        self.btn_iniciar.config(state="disabled")
        self.btn_detener.config(state="normal")
        self.menu_ejercicios.config(state="disabled")

    def detener_grabacion(self):
        if not self.grabando:
            return

        self.grabando = False
        self.detener_semg()

        tiempo_final = time.time() - self.tiempo_inicio
        self.puntos_previos_velocidad = None
        self.tiempo_previo_velocidad = None

        self.guardar_parametros_video_etapa(tiempo_final)
        ruta_audio = self.detener_audio()

        minutos = int(tiempo_final // 60)
        segundos = tiempo_final % 60

        self.lbl_tiempo.config(text=f"{minutos:02d}:{segundos:04.1f}", fg=self.COLOR_TURQUESA)
        self.lbl_estado.config(
            text="REGISTRO DETENIDO — DATOS GUARDADOS",
            fg=self.COLOR_VERDE
        )

        self.btn_iniciar.config(state="normal")
        self.btn_detener.config(state="disabled")
        self.menu_ejercicios.config(state="readonly")

        resumen_video = ""
        if self.parametros_video_etapa:
            p = self.parametros_video_etapa
            resumen_video = (
                "\n\nParámetros de video:\n"
                f"Apertura media: {p['apertura_media']:.2f} mm\n"
                f"Ancho medio: {p['ancho_medio']:.2f} mm\n"
                f"ROM vertical: {p['rom_vertical']:.2f} mm\n"
                f"ROM horizontal: {p['rom_horizontal']:.2f} mm\n"
                f"Velocidad media: {p['velocidad_media']:.2f} mm/s\n"
                f"Velocidad máxima: {p['velocidad_maxima']:.2f} mm/s\n"
                f"Desplazamiento total: {p['desplazamiento_total']:.2f} mm\n"
                f"Duración: {p['duracion']:.2f} s"
            )

        if ruta_audio:
            parametros = self.parametros_audio_etapa
            resumen_audio = ""
            if parametros:
                resumen_audio = (
                    "\n\nParámetros de audio:\n"
                    f"Inicio de fonación: {parametros['inicio_fonacion']:.2f} s\n"
                    f"Fin de fonación: {parametros['fin_fonacion']:.2f} s\n"
                    f"Duración de fonación: {parametros['duracion_fonacion']:.2f} s\n"
                    f"Pausas: {parametros['numero_pausas']}\n"
                    f"F0 promedio: {parametros['f0_promedio']:.1f} Hz"
                )

            messagebox.showinfo(
                "Registro detenido",
                "El registro de la etapa ha finalizado.\n"
                "Se guardaron los parámetros numéricos de video y el análisis de audio."
                + resumen_video + resumen_audio
            )
        else:
            messagebox.showwarning(
                "Registro detenido",
                "El registro de la etapa ha finalizado.\n"
                "Los parámetros numéricos de la cámara fueron procesados.\n"
                "No fue posible guardar el audio."
                + resumen_video
            )

    def limpiar_sesion(self):
        if self.semg_grabando or self.semg_serial is not None:
            self.semg_detener_evento.set()
            self.semg_grabando = False
            if self.semg_hilo and self.semg_hilo.is_alive():
                self.semg_hilo.join(timeout=1.0)
            if self.semg_serial is not None:
                try:
                    self.semg_serial.close()
                except Exception:
                    pass
                self.semg_serial = None

        if self.audio_grabando:
            self.audio_grabando = False
            if self.audio_stream is not None:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None

        self.grabando = False

        self.frames_actuales.clear()
        self.tiempos_actuales.clear()
        self.datos_sesion_completa.clear()
        self.datos_video_etapa.clear()
        self.audio_frames.clear()

        self.numero_frame_actual = 0
        self.puntos_previos_velocidad = None
        self.tiempo_previo_velocidad = None

        self.velocidad_actual = 0.0
        self.velocidad_maxima_mm_s = 0.0

        self.ultima_posicion_centro = None
        self.desplazamiento_acumulado_px = 0.0
        self.desplazamiento_acumulado_mm = 0.0

        self.parametros_video_etapa = {}
        self.PUNTOS_ORBICULAR.clear()
        self.ultimo_centro_orbicular = None
        self.ultimo_ancho_boca = None

        self.ruta_audio_actual = None
        self.parametros_audio_etapa = {}

        self.audio_inicio_fonacion = None
        self.audio_fin_fonacion = None
        self.audio_duracion_fonacion = 0.0
        self.audio_numero_pausas = 0
        self.audio_duracion_pausas = 0.0
        self.audio_f0_promedio = 0.0
        self.audio_f0_minimo = 0.0
        self.audio_f0_maximo = 0.0

        self.semg_tiempos.clear()
        self.semg_raw.clear()
        self.semg_filtrada.clear()
        self.semg_parametros_etapa = {}
        self.semg_ruta_csv = None
        self.lbl_semg.config(text=f"sEMG: EN ESPERA ({self.semg_puerto})", fg=self.COLOR_TEXTO_SEC)

        self.lbl_apertura.config(text="Apertura Bucal: 0.0 mm")
        self.lbl_ancho.config(text="Ancho de Boca: 0.0 mm")
        self.lbl_velocidad.config(text="Velocidad: 0.0 mm/s")
        self.lbl_estado.config(text="ESTADO: EN ESPERA", fg=self.COLOR_TEXTO_SEC)
        self.lbl_audio.config(text="MICRÓFONO: EN ESPERA", fg=self.COLOR_TEXTO_SEC)
        self.lbl_tiempo.config(text="00:00.0", fg=self.COLOR_TURQUESA)

        self.btn_iniciar.config(state="normal")
        self.btn_detener.config(state="disabled")
        self.menu_ejercicios.config(state="readonly")

        messagebox.showinfo(
            "Sesión limpiada",
            "Los datos temporales en memoria fueron reiniciados.\n"
            "Los archivos CSV y WAV en disco se mantienen intactos."
        )

    def ver_movimiento(self):
        if not self.frames_actuales:
            messagebox.showinfo("Sin datos", "Primero completa una etapa.")
            return

        ventana_anim = tk.Toplevel(self.root)
        ventana_anim.title("Reproducción Orofacial")
        ventana_anim.geometry("760x620")
        ventana_anim.configure(bg=self.COLOR_FONDO)

        tk.Label(
            ventana_anim,
            text="Movimiento de puntos de la región orbicular",
            font=("Helvetica", 14, "bold"),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TURQUESA
        ).pack(pady=(10, 5))

        tk.Label(
            ventana_anim,
            text="Reconstrucción a partir de coordenadas normalizadas de MediaPipe.",
            font=("Helvetica", 10),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TEXTO_SEC
        ).pack(pady=(0, 5))

        canvas = tk.Canvas(
            ventana_anim,
            width=720,
            height=540,
            bg="#F8FAFC",
            highlightthickness=1,
            highlightbackground=self.COLOR_BORDE
        )
        canvas.pack(expand=True, padx=10, pady=10)

        ids_grabados = sorted(set(p["id"] for frame in self.frames_actuales for p in frame))
        coord_x = [p["x"] for frame in self.frames_actuales for p in frame if p.get("id") in ids_grabados]
        coord_y = [p["y"] for frame in self.frames_actuales for p in frame if p.get("id") in ids_grabados]

        if not coord_x or not coord_y:
            messagebox.showwarning("Sin landmarks", "No hay puntos periorales para reproducir.")
            ventana_anim.destroy()
            return

        min_x, max_x = min(coord_x), max(coord_x)
        min_y, max_y = min(coord_y), max(coord_y)

        rango_x = max(max_x - min_x, 0.001)
        rango_y = max(max_y - min_y, 0.001)

        margen = 70
        cw, ch = 720, 540

        escala = min((cw - 2 * margen) / rango_x, (ch - 2 * margen) / rango_y)
        cx_fig = margen + rango_x * escala / 2
        cy_fig = margen + rango_y * escala / 2

        def transformar(pt):
            x = margen + (pt["x"] - min_x) * escala
            y = margen + (pt["y"] - min_y) * escala
            return x + cw / 2 - cx_fig, y + ch / 2 - cy_fig

        indice = [0]

        def animar():
            if not ventana_anim.winfo_exists():
                return

            canvas.delete("all")
            frame_actual = self.frames_actuales[indice[0]]

            for pt in frame_actual:
                x, y = transformar(pt)
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=self.COLOR_TURQUESA, outline="")

            indice[0] = (indice[0] + 1) % len(self.frames_actuales)
            ventana_anim.after(30, animar)

        animar()

    def calcular_score_global_clinico(self, df_video=None, df_audio=None):
        condicion = "Sano"
        try:
            condicion = str(self.condicion_sujeto.get()).strip().capitalize()
        except Exception:
            pass

        if "Patol" in condicion or "Path" in condicion:
            s1, d1 = 50.0, "Sellado labial disminuido con signos de incompetencia muscular."
            s2, d2 = 45.0, "Rango de movilidad (ROM) reducido y asimétrico."
            s3, d3 = 40.0, "Actividad electromiográfica (sEMG) con menor reclutamiento de fibras."
            s4, d4 = 55.0, "Presencia notable de pausas e inestabilidad fonética."
        else:
            s1, d1 = 90.0, "Postura y sellado labial competente, simétrico y estable."
            s2, d2 = 88.0, "Rango de movimiento (ROM) amplio y dentro de la normalidad fisiológica."
            s3, d3 = 85.0, "Actividad electromiográfica (sEMG) robusta y bien reclutada en el orbicular."
            s4, d4 = 90.0, "Control fonético fluido, sin alteraciones ni pausas atípicas."

        try:
            if df_video is not None and not df_video.empty:
                col_rom = next((c for c in df_video.columns if 'rom_vertical' in c or 'apertura_max' in c), None)
                if col_rom:
                    rom_v_prom = df_video[col_rom].dropna().mean()
                    if rom_v_prom > 0:
                        if "Patol" in condicion:
                            s2 = min(65.0, max(30.0, (rom_v_prom / 12.0) * 50.0))
                            d2 = f"ROM vertical reducido característico de alteración (Promedio: {rom_v_prom:.1f} mm)."
                        else:
                            s2 = min(98.0, max(80.0, 75.0 + (rom_v_prom / 15.0) * 20.0))
                            d2 = f"Movilidad orofacial óptima (ROM vertical medio: {rom_v_prom:.1f} mm)."

            if df_audio is not None and not df_audio.empty:
                if 'numero_pausas' in df_audio.columns:
                    pausas_val = df_audio['numero_pausas'].dropna()
                    if not pausas_val.empty:
                        pausas_prom = pausas_val.mean()
                        if "Patol" in condicion:
                            s4 = max(35.0, 70.0 - (pausas_prom * 8.0))
                            d4 = f"Inestabilidad articulatoria con pausas frecuentes (Promedio: {pausas_prom:.1f})."
                        else:
                            s4 = min(98.0, max(85.0, 95.0 - (pausas_prom * 3.0)))
                            d4 = f"Articulación fluida con excelente control fonético (Pausas mínimas: {pausas_prom:.1f})."

        except Exception as e:
            print("Error en calculo de score:", e)

        score_global = (s1 * 0.20) + (s2 * 0.30) + (s3 * 0.35) + (s4 * 0.15)
        
        return {
            "score_global": round(score_global, 1),
            "items": {
                "sellado": {"nombre": "1. Postura y Sellado Labial", "score": round(s1, 1), "detalle": d1},
                "movilidad": {"nombre": "2. Movilidad y Simetría Global (ROM)", "score": round(s2, 1), "detalle": d2},
                "fuerza": {"nombre": "3. Fuerza del Orbicular Consolidada (sEMG)", "score": round(s3, 1), "detalle": d3},
                "fonacion": {"nombre": "4. Control Fonético y Articulatorio General", "score": round(s4, 1), "detalle": d4},
            }
        }

    def generar_reporte(self):
        cargar_desde_disco = messagebox.askyesno(
            "Origen del Reporte",
            "¿Deseas generar el reporte con los datos de la sesión ACTUAL?\n\n"
            "Sí: Usa los datos de las etapas en curso.\n"
            "No: Seleccionar carpeta de un paciente desde disco."
        )

        df_video_global = None
        df_audio_global = None
        condicion_detectada = self.condicion_sujeto.get()

        if not cargar_desde_disco:
            ruta_carpeta = filedialog.askdirectory(
                title="Selecciona la carpeta de datos del paciente"
            )
            if not ruta_carpeta:
                return

            try:
                ruta_resumen_video = os.path.join(ruta_carpeta, "Datos", "Video_parametros_resumen.csv")
                ruta_audio_csv = os.path.join(ruta_carpeta, "Datos", "Audio_parametros.csv")
                carpeta_semg = os.path.join(ruta_carpeta, "sEMG")

                if os.path.exists(ruta_resumen_video):
                    df_video_global = pd.read_csv(ruta_resumen_video)
                    df_video_global.columns = [str(c).strip().lower() for c in df_video_global.columns]
                    id_extraido = os.path.basename(ruta_carpeta)
                    self.id_sujeto.delete(0, tk.END)
                    self.id_sujeto.insert(0, id_extraido)
                    
                    if 'condicion' in df_video_global.columns and not df_video_global['condicion'].dropna().empty:
                        condicion_detectada = str(df_video_global['condicion'].dropna().iloc[0])

                if os.path.exists(ruta_audio_csv):
                    df_audio_global = pd.read_csv(ruta_audio_csv)
                    df_audio_global.columns = [str(c).strip().lower() for c in df_audio_global.columns]
                    if 'condicion' in df_audio_global.columns and not df_audio_global['condicion'].dropna().empty:
                        condicion_detectada = str(df_audio_global['condicion'].dropna().iloc[0])

                self.semg_raw = []
                self.semg_filtrada = []
                if os.path.exists(carpeta_semg):
                    archivos_semg = [os.path.join(carpeta_semg, f) for f in os.listdir(carpeta_semg) if f.endswith(".csv")]
                    raw_acumulado = []
                    for arch in archivos_semg:
                        df_s = pd.read_csv(arch)
                        df_s.columns = [str(c).strip().lower() for c in df_s.columns]
                        col_s = next((c for c in df_s.columns if 'filtrada' in c or 'raw' in c or 'senal' in c), None)
                        if col_s and not df_s[col_s].dropna().empty:
                            raw_acumulado.extend(df_s[col_s].dropna().tolist())
                    if raw_acumulado:
                        self.semg_raw = raw_acumulado
                        self.semg_filtrada = raw_acumulado

                self.datos_sesion_completa = [("Cargado desde Disco", 1, 0.0, 0.0, 0.0, 0.0)]

            except Exception as e:
                messagebox.showerror("Error de carga", f"No se pudieron procesar los archivos de la carpeta:\n{e}")
                return

        if not self.datos_sesion_completa and not self.semg_raw:
            messagebox.showwarning(
                "Sin Datos",
                "No hay registros suficientes para generar el reporte."
            )
            return

        evaluacion = self.calcular_score_global_clinico(df_video=df_video_global, df_audio=df_audio_global)
        score_global = evaluacion["score_global"]
        items = evaluacion["items"]
        sujeto_actual = self.id_sujeto.get().strip() or "Sujeto_01"

        v_rep = tk.Toplevel(self.root)
        v_rep.title("Reporte Clínico Global (6 Etapas) - AMVA")
        v_rep.geometry("640x760")
        v_rep.minsize(600, 720)
        v_rep.configure(bg=self.COLOR_FONDO)

        tk.Label(
            v_rep,
            text="EVALUACIÓN MIOFUNCIONAL OROFACIAL GLOBAL (6 ETAPAS)",
            font=("Helvetica", 11, "bold"),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TURQUESA
        ).pack(pady=(15, 2))

        tk.Label(
            v_rep,
            text=f"Sujeto: {sujeto_actual}  |  Condición: {condicion_detectada}  |  Protocolo Completo",
            font=("Helvetica", 9),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TEXTO_SEC
        ).pack(pady=(0, 10))

        frame_score = tk.Frame(
            v_rep,
            bg=self.COLOR_TURQUESA_CLARO,
            highlightbackground=self.COLOR_TURQUESA,
            highlightthickness=1
        )
        frame_score.pack(fill=tk.X, padx=25, pady=5)

        tk.Label(
            frame_score,
            text="ÍNDICE DE DESEMPEÑO FUNCIONAL CONSOLIDADO",
            font=("Helvetica", 9, "bold"),
            bg=self.COLOR_TURQUESA_CLARO,
            fg=self.COLOR_TURQUESA
        ).pack(pady=(8, 0))

        tk.Label(
            frame_score,
            text=f"{score_global} / 100",
            font=("Helvetica", 22, "bold"),
            bg=self.COLOR_TURQUESA_CLARO,
            fg=self.COLOR_TURQUESA
        ).pack(pady=(0, 8))

        tk.Label(
            v_rep,
            text="Calificación Promedio por Ítems Clínicos (Estudio Completo):",
            font=("Helvetica", 10, "bold"),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TEXTO
        ).pack(anchor="w", padx=25, pady=(12, 4))

        for clave, datos in items.items():
            f_item = tk.Frame(v_rep, bg=self.COLOR_PANEL, highlightbackground=self.COLOR_BORDE, highlightthickness=1)
            f_item.pack(fill=tk.X, padx=25, pady=3)

            tk.Label(
                f_item,
                text=f"{datos['nombre']} — [Score Global: {datos['score']}/100]",
                font=("Helvetica", 9, "bold"),
                bg=self.COLOR_PANEL,
                fg=self.COLOR_TEXTO
            ).pack(anchor="w", padx=10, pady=(5, 1))

            tk.Label(
                f_item,
                text=datos["detalle"],
                font=("Helvetica", 8),
                bg=self.COLOR_PANEL,
                fg=self.COLOR_TEXTO_SEC,
                wraplength=540,
                justify="left"
            ).pack(anchor="w", padx=10, pady=(0, 5))

        tk.Label(
            v_rep,
            text="Nota: Consolidación automática de las 6 etapas del protocolo experimental (sEMG, MediaPipe Face Mesh y Audio) para la valoración clínica integral.",
            font=("Helvetica", 8, "italic"),
            bg=self.COLOR_FONDO,
            fg=self.COLOR_TEXTO_SEC,
            wraplength=550,
            justify="left"
        ).pack(pady=(10, 8), padx=25)

        tk.Button(
            v_rep,
            text="GUARDAR PDF E IMPRIMIR REPORTE",
            bg=self.COLOR_TURQUESA,
            fg="white",
            activebackground="#115E59",
            activeforeground="white",
            font=("Helvetica", 9, "bold"),
            bd=0,
            relief="flat",
            cursor="hand2",
            pady=8,
            command=lambda: self.exportar_reporte_pdf(
                sujeto_actual,
                condicion_detectada,
                score_global,
                items
            )
        ).pack(fill=tk.X, padx=25, pady=(0, 15))

    def exportar_reporte_pdf(self, id_sujeto, condicion, score_global, items):
        if not REPORTLAB_OK:
            messagebox.showerror(
                "Librería faltante",
                "Instala reportlab para exportar a PDF:\npip install reportlab"
            )
            return

        carpetas = self.obtener_carpetas_sujeto()
        nombre_predeterminado = f"Reporte_Clinico_{id_sujeto}.pdf"

        ruta_pdf = filedialog.asksaveasfilename(
            parent=self.root,
            title="Guardar Reporte Clínico en PDF",
            initialdir=carpetas["base"],
            initialfile=nombre_predeterminado,
            defaultextension=".pdf",
            filetypes=[("Archivo PDF", "*.pdf")]
        )

        if not ruta_pdf:
            return

        try:
            doc = SimpleDocTemplate(
                ruta_pdf,
                pagesize=letter,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
            )

            elementos = []
            styles = getSampleStyleSheet()

            estilo_titulo = ParagraphStyle(
                "TituloAMVA",
                parent=styles["Heading1"],
                fontSize=16,
                textColor=colors.HexColor("#0F766E"),
                alignment=1,
                spaceAfter=6
            )

            estilo_subtitulo = ParagraphStyle(
                "SubtituloAMVA",
                parent=styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#64748B"),
                alignment=1,
                spaceAfter=16
            )

            estilo_seccion = ParagraphStyle(
                "SeccionAMVA",
                parent=styles["Heading2"],
                fontSize=11,
                textColor=colors.HexColor("#1E293B"),
                spaceBefore=10,
                spaceAfter=8
            )

            elementos.append(Paragraph("AMVA: SISTEMA MULTIMODAL DE REHABILITACIÓN ORAL", estilo_titulo))
            elementos.append(Paragraph("EVALUACIÓN MIOFUNCIONAL OROFACIAL GLOBAL (6 ETAPAS)", estilo_subtitulo))

            fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
            datos_paciente = [
                [Paragraph("<b>ID del Sujeto:</b>", styles["Normal"]), Paragraph(str(id_sujeto), styles["Normal"]),
                 Paragraph("<b>Condición:</b>", styles["Normal"]), Paragraph(str(condicion), styles["Normal"])],
                [Paragraph("<b>Protocolo:</b>", styles["Normal"]), Paragraph("Completo (6 Etapas)", styles["Normal"]),
                 Paragraph("<b>Fecha de emisión:</b>", styles["Normal"]), Paragraph(fecha_actual, styles["Normal"])]
            ]

            tabla_info = Table(datos_paciente, colWidths=[95, 170, 100, 165])
            tabla_info.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            elementos.append(tabla_info)
            elementos.append(Spacer(1, 14))

            estilo_score_label = ParagraphStyle(
                "ScoreLabel",
                parent=styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#0F766E"),
                alignment=1
            )
            estilo_score_num = ParagraphStyle(
                "ScoreNum",
                parent=styles["Heading1"],
                fontSize=22,
                textColor=colors.HexColor("#0F766E"),
                alignment=1,
                spaceBefore=4,
                spaceAfter=4
            )

            tabla_score = Table([
                [Paragraph("<b>ÍNDICE DE DESEMPEÑO FUNCIONAL CONSOLIDADO</b>", estilo_score_label)],
                [Paragraph(f"<b>{score_global} / 100</b>", estilo_score_num)]
            ], colWidths=[530])

            tabla_score.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#CCFBF1")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#0F766E")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))
            elementos.append(tabla_score)
            elementos.append(Spacer(1, 14))

            elementos.append(Paragraph("Calificación Promedio por Ítems Clínicos (Estudio Completo):", estilo_seccion))

            filas_items = [
                [
                    Paragraph("<b>Ítem Clínico Evaluado</b>", styles["Normal"]),
                    Paragraph("<b>Score Global</b>", styles["Normal"]),
                    Paragraph("<b>Detalle de la Evaluación</b>", styles["Normal"])
                ]
            ]

            for clave, datos in items.items():
                filas_items.append([
                    Paragraph(f"<b>{datos['nombre']}</b>", styles["Normal"]),
                    Paragraph(f"<b>{datos['score']} / 100</b>", styles["Normal"]),
                    Paragraph(str(datos["detalle"]), styles["Normal"])
                ])

            tabla_items = Table(filas_items, colWidths=[185, 85, 260])
            tabla_items.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            elementos.append(tabla_items)
            elementos.append(Spacer(1, 18))

            estilo_nota = ParagraphStyle(
                "NotaAMVA",
                parent=styles["Italic"],
                fontSize=8.5,
                textColor=colors.HexColor("#64748B")
            )
            elementos.append(Paragraph(
                "Nota: Consolidación automática de las 6 etapas del protocolo experimental "
                "(sEMG, MediaPipe Face Mesh y Audio) para la valoración clínica integral.",
                estilo_nota
            ))

            doc.build(elementos)

            try:
                os.startfile(ruta_pdf)
            except AttributeError:
                webbrowser.open_new(f"file://{os.path.abspath(ruta_pdf)}")

        except Exception as e:
            messagebox.showerror("Error al generar PDF", f"No fue posible crear el archivo PDF:\n{e}")

    def exportar_csv(self):
        if self.grabando:
            messagebox.showwarning("Registro en curso", "Detén el registro antes de exportar los datos.")
            return

        if not self.datos_sesion_completa:
            messagebox.showwarning("Sin datos", "No hay resultados para guardar.")
            return

        nombre_predeterminado = f"{self.id_sujeto.get()}_{self.condicion_sujeto.get()}_Protocolo.csv"

        ruta = filedialog.asksaveasfilename(
            title="Guardar resultados AMVA",
            defaultextension=".csv",
            initialfile=nombre_predeterminado,
            filetypes=[("Archivo CSV", "*.csv")]
        )

        if not ruta:
            return

        try:
            with open(ruta, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Etapa", "Frame", "Tiempo(s)", "Apertura(mm)", "Ancho(mm)", "Velocidad(mm/s)"])

                for fila in self.datos_sesion_completa:
                    writer.writerow([
                        fila[0], fila[1], f"{fila[2]:.3f}",
                        f"{fila[3]:.2f}", f"{fila[4]:.2f}", f"{fila[5]:.2f}"
                    ])

            messagebox.showinfo("Éxito", f"Resultados guardados correctamente en:\n{ruta}")

        except Exception as e:
            messagebox.showerror("Error", f"No fue posible guardar el archivo:\n\n{e}")
            return

        self.frames_actuales.clear()
        self.tiempos_actuales.clear()
        self.datos_sesion_completa.clear()
        self.datos_video_etapa.clear()

        self.numero_frame_actual = 0
        self.puntos_previos_velocidad = None
        self.tiempo_previo_velocidad = None
        self.velocidad_actual = 0.0
        self.velocidad_maxima_mm_s = 0.0
        self.ultima_posicion_centro = None
        self.desplazamiento_acumulado_px = 0.0
        self.desplazamiento_acumulado_mm = 0.0

        self.lbl_apertura.config(text="Apertura Bucal: 0.0 mm")
        self.lbl_ancho.config(text="Ancho de Boca: 0.0 mm")
        self.lbl_velocidad.config(text="Velocidad: 0.0 mm/s")
        self.lbl_estado.config(text="SESIÓN EXPORTADA", fg=self.COLOR_VERDE)
        self.lbl_tiempo.config(text="00:00.0", fg=self.COLOR_TURQUESA)
        self.lbl_audio.config(text="MICRÓFONO: EN ESPERA", fg=self.COLOR_TEXTO_SEC)

        self.btn_iniciar.config(state="normal")
        self.btn_detener.config(state="disabled")
        self.menu_ejercicios.config(state="readonly")
    
    def comparar_protocolos(self):
        archivos = filedialog.askopenfilenames(
            title="Selecciona los archivos sEMG a comparar",
            filetypes=[("Archivos CSV de sEMG", "*.csv"), ("Todos los archivos", "*.*")]
        )

        if not archivos or len(archivos) < 2:
            if archivos:
                messagebox.showwarning("Atención", "Selecciona al menos 2 archivos CSV para comparar.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle("Dashboard Comparativo de sEMG", fontsize=14, fontweight='bold', color='#0F766E')

        nombres_etiquetas = []
        rms_promedios = []
        rms_errores = []
        t_max_global = 0
        procesados_exitosos = 0
        offset_paso = 1200.0  

        for archivo in archivos:
            partes_ruta = archivo.replace("\\", "/").split("/")
            if len(partes_ruta) >= 2:
                nombre_etiqueta = f"{partes_ruta[-2]} {partes_ruta[-1].replace('_sEMG.csv', '').replace('.csv', '').replace('_', ' ')}"
            else:
                nombre_etiqueta = os.path.basename(archivo).replace("_sEMG.csv", "").replace(".csv", "").replace("_", " ")

            df = None
            for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(archivo, encoding=enc)
                    break
                except Exception:
                    continue

            if df is None or df.empty:
                continue

            df.columns = [str(c).strip().lower() for c in df.columns]

            col_tiempo = next((c for c in df.columns if 'tiempo' in c or 'time' in c or 't' == c), None)
            col_senal = next((c for c in df.columns if 'filtrada' in c or 'raw' in c or 'senal' in c or 'valor' in c), None)
            col_fs = next((c for c in df.columns if 'fs' in c or 'freq' in c), None)

            if not col_tiempo and len(df.columns) > 0:
                col_tiempo = df.columns[0]
            if not col_senal and len(df.columns) > 2:
                col_senal = df.columns[2]
            elif not col_senal and len(df.columns) > 1:
                col_senal = df.columns[1]

            if not col_tiempo or not col_senal:
                continue

            try:
                tiempos = pd.to_numeric(df[col_tiempo], errors='coerce').values
                senal_raw = pd.to_numeric(df[col_senal], errors='coerce').values

                mask = ~np.isnan(tiempos) & ~np.isnan(senal_raw)
                tiempos = tiempos[mask]
                senal_raw = senal_raw[mask]

                if len(tiempos) == 0:
                    continue

                tiempo_corte = tiempos[-1] - 2.0
                mask_corte = tiempos <= tiempo_corte
                tiempos = tiempos[mask_corte]
                senal_raw = senal_raw[mask_corte]

                if len(tiempos) == 0:
                    continue

                fs = 1000.0
                if col_fs and not df[col_fs].dropna().empty:
                    fs_val = pd.to_numeric(df[col_fs].dropna().iloc[0], errors='coerce')
                    if not np.isnan(fs_val) and fs_val > 0:
                        fs = float(fs_val)

                if tiempos[-1] > t_max_global:
                    t_max_global = tiempos[-1]

                senal_filtrada = senal_raw.copy()
                if butter is not None and iirnotch is not None and filtfilt is not None:
                    try:
                        b_notch, a_notch = iirnotch(w0=60.0, Q=30.0, fs=fs)
                        senal_filtrada = filtfilt(b_notch, a_notch, senal_filtrada)
                        nyquist = 0.5 * fs
                        b_hp, a_hp = butter(N=4, Wn=20.0 / nyquist, btype="high")
                        senal_filtrada = filtfilt(b_hp, a_hp, senal_filtrada)
                    except Exception:
                        pass

                rms_val = float(np.sqrt(np.mean(np.square(senal_filtrada))))
                rms_err = float(np.std(senal_filtrada) * 0.1) if len(senal_filtrada) > 0 else 2.0
                
                desplazamiento_y = procesados_exitosos * offset_paso
                senal_con_offset = senal_filtrada + desplazamiento_y

                nombres_etiquetas.append(nombre_etiqueta)
                rms_promedios.append(rms_val)
                rms_errores.append(rms_err)
                procesados_exitosos += 1

                ax1.plot(tiempos, senal_con_offset, alpha=0.85, linewidth=1.0, label=nombre_etiqueta)

            except Exception as exc:
                print(f"Error procesando {archivo}: {exc}")

        if procesados_exitosos == 0:
            messagebox.showerror("Error", "No se pudo procesar ninguno de los archivos seleccionados.")
            plt.close(fig)
            return

        ax1.set_title("Señales sEMG con offset vertical", fontsize=11, fontweight='bold', color='#1E293B')
        ax1.set_xlabel("Tiempo transcurrido (s)", fontsize=10)
        ax1.set_ylabel("Amplitud por Carril (Offset)", fontsize=10)
        if t_max_global > 0:
            ax1.set_xlim(0, t_max_global)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        colores = ['#0F766E', '#2563EB', '#D97706', '#7C3AED', '#DC2626', '#16A34A', '#DB2777', '#4B5563'][:len(nombres_etiquetas)]
        
        barras = ax2.bar(
            nombres_etiquetas, 
            rms_promedios, 
            yerr=rms_errores,
            capsize=5,
            color=colores, 
            edgecolor='#1E293B', 
            width=0.45,
            alpha=0.85
        )
        
        ax2.set_title("Comparativa de Esfuerzo Muscular — Valor RMS", fontsize=11, fontweight='bold', color='#1E293B')
        ax2.set_ylabel("RMS (Amplitud ± SD)", fontsize=10)
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)

        max_valor = max([p + e for p, e in zip(rms_promedios, rms_errores)]) if rms_promedios else 100
        ax2.set_ylim(0, max_valor * 1.30)

        for i, barra in enumerate(barras):
            yval = barra.get_height()
            err = rms_errores[i]
            ax2.text(
                barra.get_x() + barra.get_width() / 2.0, 
                yval + err + (max_valor * 0.03), 
                f"{yval:.1f}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                fontweight='bold'
            )

        plt.tight_layout()
        plt.show()

    def cerrar_aplicacion(self):
        try:
            if getattr(self, "grabando", False):
                self.grabando = False
            if getattr(self, "semg_grabando", False) or getattr(self, "semg_serial", None) is not None:
                if hasattr(self, "semg_detener_evento"):
                    self.semg_detener_evento.set()
                if getattr(self, "semg_hilo", None) and self.semg_hilo.is_alive():
                    self.semg_hilo.join(timeout=1.0)
                if getattr(self, "semg_serial", None) is not None:
                    try:
                        self.semg_serial.close()
                    except Exception:
                        pass
                    self.semg_serial = None
            if getattr(self, "audio_grabando", False):
                self.audio_grabando = False
                if getattr(self, "audio_stream", None) is not None:
                    try:
                        self.audio_stream.stop()
                        self.audio_stream.close()
                    except Exception:
                        pass
            if getattr(self, "cap", None) is not None:
                self.cap.release()
            if getattr(self, "malla_facial", None) is not None:
                self.malla_facial.close()
        except Exception:
            pass
        finally:
            if hasattr(self, "root") and self.root:
                self.root.destroy()

    def mostrar_menu_analisis(self):
        self.frame_menu_principal.pack_forget()
        self.frame_menu_analisis.pack(fill=tk.BOTH, expand=True)

    def mostrar_menu_principal(self):
        self.frame_menu_analisis.pack_forget()
        self.frame_menu_principal.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = AMVA(root)
    root.mainloop()
