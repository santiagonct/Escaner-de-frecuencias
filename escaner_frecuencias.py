"""Escáner de frecuencias en tiempo real.

Programa corto para visualizar un espectrograma y sus picos usando una entrada de audio en tiempo real. Obteniendo una nota musical de cada frecuencia y guardando el resultado en un archivo CSV.
"""

import argparse
import queue
import threading
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks, windows
import sounddevice as sd

# ── Configuración ──────────────────────────────────────────────────────────
TASA_MUESTREO    = 44100
TAM_BLOQUE       = 4096
NUM_PICOS        = 4
FREQ_MIN_HZ      = 60
FREQ_MAX_HZ      = 8000
PROMINENCIA_MIN  = 0.06
HISTORIAL_SEG    = 6
FPS              = 10        
ARCHIVO_DB       = "frecuencias_registradas.csv"
# ──────────────────────────────────────────────────────────────────────────

NOMBRES_NOTAS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def hz_a_nota(freq: float) -> tuple[str, float]:
    if freq <= 0:
        return "—", 0.0
    midi_f = 12 * np.log2(freq / 440) + 69
    midi_r = int(round(midi_f))
    midi_r = max(0, min(127, midi_r))
    nombre = f"{NOMBRES_NOTAS[midi_r % 12]}{midi_r // 12 - 1}"
    cents  = (midi_f - midi_r) * 100
    return nombre, round(cents, 1)


class EscanerFrecuencias:
    def __init__(self, dispositivo=None, duracion=None):
        self.dispositivo   = dispositivo
        self.duracion      = duracion
        self.cola          = queue.Queue(maxsize=4)  
        self.ejecutando    = True
        self.transcurrido  = 0.0
        self._datos_nuevos = False  

        self.señal_buf     = np.zeros(TAM_BLOQUE)
        freqs_todas        = np.fft.rfftfreq(TAM_BLOQUE, d=1.0 / TASA_MUESTREO)
        self.mascara_freq  = (freqs_todas >= FREQ_MIN_HZ) & (freqs_todas <= FREQ_MAX_HZ)
        self.frecuencias_graf = freqs_todas[self.mascara_freq]
        N_FREQ             = self.mascara_freq.sum()
        self.magnitudes_graf = np.zeros(N_FREQ)
        self.picos          = []

        pasos = int(HISTORIAL_SEG * TASA_MUESTREO / TAM_BLOQUE)
        self.hist_espectro = np.full((N_FREQ, pasos), -80.0)

        self._hann = windows.hann(TAM_BLOQUE)

        self._iniciar_csv()

    def _iniciar_csv(self):
        existe = Path(ARCHIVO_DB).exists()
        self._archivo_csv = open(ARCHIVO_DB, "a", newline="", encoding="utf-8")
        self._escritor_csv = csv.DictWriter(self._archivo_csv, fieldnames=[
            "timestamp","elapsed_s","frecuencia_hz",
            "nota","cents_desafinacion","magnitud_norm","rango",
        ])
        if not existe:
            self._escritor_csv.writeheader()

    def _guardar_picos(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for p in self.picos:
            self._escritor_csv.writerow({
                "timestamp":          ts,
                "elapsed_s":          round(self.transcurrido, 2),
                "frecuencia_hz":      round(p["hz"], 2),
                "nota":               p["nota"],
                "cents_desafinacion": p["cents"],
                "magnitud_norm":      round(p["mag"], 4),
                "rango":              p["rango"],
            })
        self._archivo_csv.flush()

    def _procesar(self, bloque: np.ndarray):
        self.señal_buf = bloque

        Y         = np.fft.rfft(bloque * self._hann)
        mags_full = np.abs(Y) * (2.0 / TAM_BLOQUE)
        mags_full[0] /= 2

        self.hist_espectro = np.roll(self.hist_espectro, -1, axis=1)
        self.hist_espectro[:, -1] = 20 * np.log10(
            mags_full[self.mascara_freq] + 1e-10)

        m  = mags_full[self.mascara_freq]
        mx = m.max()
        if mx > 0:
            m = m / mx
        self.magnitudes_graf = m

        indices, props = find_peaks(m, prominence=PROMINENCIA_MIN, distance=5)
        picos = []
        if len(indices):
            top = np.argsort(m[indices])[::-1][:NUM_PICOS]
            for rango, io in enumerate(top, 1):
                i = indices[io]
                nota, cents = hz_a_nota(float(self.frecuencias_graf[i]))
                picos.append({
                    "hz":    float(self.frecuencias_graf[i]),
                    "mag":   float(m[i]),
                    "prom":  float(props["prominences"][io]),
                    "nota":  nota,
                    "cents": cents,
                    "rango": rango,
                })
        self.picos       = picos
        self._datos_nuevos = True

    def _cb_audio(self, indata, frames, time, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        if not self.cola.full():
            self.cola.put(mono.copy())

    def _trabajador(self):
        bloques    = 0
        save_cada  = max(1, int(TASA_MUESTREO / TAM_BLOQUE))
        while self.ejecutando:
            try:
                bloque = self.cola.get(timeout=0.5)
            except queue.Empty:
                continue
            self._procesar(bloque)
            bloques         += 1
            self.transcurrido = bloques * TAM_BLOQUE / TASA_MUESTREO
            if bloques % save_cada == 0 and self.picos:
                self._guardar_picos()
            if self.duracion and self.transcurrido >= self.duracion:
                self.ejecutando = False

    def _construir_ui(self):
        BG   = "#08080f"
        GRID = "#12121f"
        SP   = "#1e2840"
        TXT  = "#b8cce8"
        ACC  = "#2a8fff"
        ORG  = "#ff5c2a"
        YEL  = "#ffcc55"
        self._c = dict(bg=BG,txt=TXT,acc=ACC,org=ORG,yel=YEL,grid=GRID,sp=SP)

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(13, 8), facecolor=BG)
        fig.canvas.manager.set_window_title("Escáner de frecuencias — tiempo real")

        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[0.7, 1.3, 1.8],
                               hspace=0.55, wspace=0.35,
                               top=0.88, bottom=0.07,
                               left=0.07, right=0.97)

        fig.text(0.5, 0.95, "ESCÁNER DE FRECUENCIAS  —  TIEMPO REAL",
                 ha="center", fontsize=13, fontweight="bold",
                 color=TXT, fontfamily="monospace")
        self._lbl_st = fig.text(0.5, 0.915, "iniciando...",
                                ha="center", fontsize=8.5,
                                color=TXT, alpha=0.5,
                                fontfamily="monospace")

        def estilo(ax, titulo):
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(SP); sp.set_linewidth(0.7)
            ax.tick_params(colors=TXT, labelsize=7)
            ax.yaxis.label.set_color(TXT)
            ax.xaxis.label.set_color(TXT)
            ax.set_title(titulo, color=TXT, fontsize=8,
                         fontfamily="monospace", pad=5, loc="left")
            ax.grid(color=GRID, linewidth=0.5)

        ax_w = fig.add_subplot(gs[0, :])
        t_ms = np.linspace(0, TAM_BLOQUE / TASA_MUESTREO * 1000, TAM_BLOQUE)
        self._ln_wave, = ax_w.plot(t_ms, np.zeros(TAM_BLOQUE),
                                   color=ACC, linewidth=0.5, alpha=0.8)
        ax_w.set_xlim(0, t_ms[-1])
        ax_w.set_ylim(-1, 1)
        ax_w.set_xlabel("ms", fontsize=7)
        ax_w.set_ylabel("Amplitud", fontsize=7)
        estilo(ax_w, "▸ FORMA DE ONDA")
        self._ax_w = ax_w

        ax_f = fig.add_subplot(gs[1, 0])
        N    = len(self.frecuencias_graf)
        segs = [[(x, 0), (x, 0)] for x in self.frecuencias_graf]
        self._lc_fft = LineCollection(segs, colors=ACC,
                                      linewidths=0.7, alpha=0.7)
        ax_f.add_collection(self._lc_fft)
       
        self._ln_fft, = ax_f.plot(self.frecuencias_graf, np.zeros(N),
                                  color=ACC, linewidth=0.8, alpha=0.9)
        
        self._sc_pk = ax_f.scatter([], [], color=ORG, s=55,
                                   zorder=5, edgecolors=YEL, linewidths=0.8)
        ax_f.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
        ax_f.set_ylim(0, 1.2)
        ax_f.set_xlabel("Frecuencia (Hz)", fontsize=7)
        ax_f.set_ylabel("Magnitud", fontsize=7)
        estilo(ax_f, "▸ ESPECTRO FFT")
        self._ax_f = ax_f

        self._ann_texts = []
        for _ in range(NUM_PICOS):
            t = ax_f.text(0, 0, "", fontsize=6.5, color=YEL,
                          fontfamily="monospace", ha="center",
                          va="bottom", visible=False)
            self._ann_texts.append(t)

        ax_n = fig.add_subplot(gs[1, 1])
        ax_n.set_facecolor(BG)
        ax_n.set_xlim(0, 1); ax_n.set_ylim(0, 1)
        ax_n.axis("off")
        ax_n.set_title("▸ NOTAS DETECTADAS", color=TXT, fontsize=8,
                       fontfamily="monospace", pad=5, loc="left")
        self._ax_n = ax_n

        col_x   = [0.04, 0.35, 0.60, 0.80]
        col_hdr = ["Hz", "Nota", "Cents", "Mag"]
        self._hdr_texts = []
        for h, x in zip(col_hdr, col_x):
            t = ax_n.text(x, 0.88, h, color=TXT, fontsize=7.5,
                          fontweight="bold", fontfamily="monospace",
                          transform=ax_n.transAxes)
            self._hdr_texts.append(t)

        ROW_COLS = [ORG, "#ff8855", "#ffaa44", YEL]
        self._row_texts = []  
        self._row_bars  = []   
        for row in range(NUM_PICOS):
            y   = 0.88 - 0.20 * (row + 1)
            col = ROW_COLS[min(row, len(ROW_COLS)-1)]
            fila = []
            for x in col_x:
                t = ax_n.text(x, y, "", color=col, fontsize=8.5,
                              fontfamily="monospace",
                              transform=ax_n.transAxes, visible=False)
                fila.append(t)
            self._row_texts.append(fila)
            bar = plt.Rectangle((0.60, y - 0.04), 0, 0.055,
                                 color=col, alpha=0.25,
                                 transform=ax_n.transAxes)
            ax_n.add_patch(bar)
            self._row_bars.append(bar)

        self._no_signal_txt = ax_n.text(
            0.5, 0.5, "sin señal", ha="center", va="center",
            color=TXT, alpha=0.35, fontsize=9,
            fontfamily="monospace", transform=ax_n.transAxes, visible=True)

        ax_s = fig.add_subplot(gs[2, :])
        t_ax = np.linspace(-HISTORIAL_SEG, 0, self.hist_espectro.shape[1])
        self._im = ax_s.pcolormesh(t_ax, self.frecuencias_graf, self.hist_espectro,
                                   cmap="inferno", shading="gouraud",
                                   vmin=-80, vmax=0)
        cb = fig.colorbar(self._im, ax=ax_s, pad=0.01, fraction=0.015)
        cb.ax.tick_params(colors=TXT, labelsize=6)
        cb.set_label("dB", color=TXT, fontsize=7)
        cb.outline.set_edgecolor(SP)
        ax_s.set_xlim(-HISTORIAL_SEG, 0)
        ax_s.set_ylim(FREQ_MIN_HZ, FREQ_MAX_HZ)
        ax_s.set_xlabel("Tiempo (s)", fontsize=7)
        ax_s.set_ylabel("Frecuencia (Hz)", fontsize=7)
        estilo(ax_s, f"▸ ESPECTROGRAMA  ({HISTORIAL_SEG}s)")
        self._ax_s = ax_s

        self._pk_hlines = []
        for _ in range(NUM_PICOS):
            ln = ax_s.axhline(-1, color=ORG, linewidth=0.55,
                              alpha=0.4, linestyle="--", visible=False)
            self._pk_hlines.append(ln)

        self._fig = fig
        self._artistas_blit = (
            [self._ln_wave, self._lc_fft, self._ln_fft,
             self._sc_pk, self._im, self._lbl_st]
            + self._ann_texts
            + [t for fila in self._row_texts for t in fila]
            + self._row_bars
            + self._pk_hlines
            + [self._no_signal_txt]
        )
        return fig

    def _actualizar(self, _frame):
        if not self._datos_nuevos:
            return self._artistas_blit

        self._datos_nuevos = False
        c = self._c

        self._ln_wave.set_ydata(self.señal_buf)
        amp = max(float(np.abs(self.señal_buf).max()), 0.01)
        self._ax_w.set_ylim(-amp * 1.15, amp * 1.15)

        mags = self.magnitudes_graf
        segs = [[(x, 0), (x, y)] for x, y in zip(self.frecuencias_graf, mags)]
        self._lc_fft.set_segments(segs)
        self._ln_fft.set_ydata(mags)

        if self.picos:
            px = np.array([p["hz"]  for p in self.picos])
            py = np.array([p["mag"] for p in self.picos])
            self._sc_pk.set_offsets(np.column_stack([px, py]))
        else:
            self._sc_pk.set_offsets(np.empty((0, 2)))

        for i, ann in enumerate(self._ann_texts):
            if i < len(self.picos):
                p = self.picos[i]
                ann.set_position((p["hz"], p["mag"] + 0.05))
                ann.set_text(f"{p['hz']:.0f}Hz")
                ann.set_visible(True)
            else:
                ann.set_visible(False)

        tiene = len(self.picos) > 0
        self._no_signal_txt.set_visible(not tiene)
        for row in range(NUM_PICOS):
            fila = self._row_texts[row]
            bar  = self._row_bars[row]
            if row < len(self.picos):
                p    = self.picos[row]
                sign = "+" if p["cents"] >= 0 else ""
                vals = [f"{p['hz']:.1f}",
                        p["nota"],
                        f"{sign}{p['cents']:.0f}¢",
                        f"{p['mag']:.3f}"]
                for col_txt, val in zip(fila, vals):
                    col_txt.set_text(val)
                    col_txt.set_visible(True)
                bar.set_width(p["mag"] * 0.34)
                bar.set_visible(True)
            else:
                for col_txt in fila:
                    col_txt.set_visible(False)
                bar.set_width(0)
                bar.set_visible(False)

        self._im.set_array(self.hist_espectro.ravel())

        for i, ln in enumerate(self._pk_hlines):
            if i < len(self.picos):
                ln.set_ydata([self.picos[i]["hz"], self.picos[i]["hz"]])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

        self._lbl_st.set_text(
            f"⏱ {self.transcurrido:.1f}s  |  "
            f"fs={TASA_MUESTREO}Hz  |  "
            f"bloque={TAM_BLOQUE}  |  "
            f"{TASA_MUESTREO/TAM_BLOQUE:.1f}Hz/bin"
        )
        return self._artistas_blit

    def ejecutar(self):
        fig = self._construir_ui()

        worker = threading.Thread(target=self._trabajador, daemon=True)
        worker.start()

        stream = sd.InputStream(
            samplerate=TASA_MUESTREO,
            blocksize=TAM_BLOQUE,
            device=self.dispositivo,
            channels=1,
            dtype="float32",
            callback=self._cb_audio,
        )

        print("\n  Escuchando... Cierra la ventana o Ctrl+C para parar.\n")
        anim = None
        with stream:
            try:
                anim = FuncAnimation(fig, self._actualizar,
                                     interval=1000 // FPS,
                                     blit=False,
                                     cache_frame_data=False)
                plt.show()
            except KeyboardInterrupt:
                pass
            finally:
                if anim is not None and getattr(anim, 'event_source', None) is not None:
                    try:
                        anim.event_source.stop()
                    except Exception:
                        pass

        self.ejecutando = False
        worker.join(timeout=2)
        self._archivo_csv.close()
        print(f"\n  Sesión guardada en: {ARCHIVO_DB}")
        print(f"  Duración: {self.transcurrido:.1f}s\n")

def main():
    parser = argparse.ArgumentParser(
        description="Escáner de frecuencias en tiempo real — optimizado")
    parser.add_argument("--device",       type=int,   default=None)
    parser.add_argument("--duration",     type=float, default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print("\nDispositivos de audio disponibles:\n")
        print(sd.query_devices())
        print()
        return

    print("\nEscáner de frecuencias — tiempo real\n")
    print(f"  Dispositivo : {'por defecto' if args.device is None else args.device}")
    print(f"  Tasa muestreo: {TASA_MUESTREO} Hz")
    print(f"  Bloque       : {TAM_BLOQUE} muestras ({TAM_BLOQUE/TASA_MUESTREO*1000:.1f} ms)")
    print(f"  Resolución   : {TASA_MUESTREO/TAM_BLOQUE:.1f} Hz/bin")
    print(f"  FPS UI       : {FPS}\n")

    EscanerFrecuencias(dispositivo=args.device, duracion=args.duration).ejecutar()

if __name__ == "__main__":
    main()