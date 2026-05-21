# Escaner de frecuencias

Herramienta de análisis de audio que detecta frecuencias dominantes usando la Transformada de Fourier y las registra en una base de datos CSV.

## ¿Qué hace?

Escuchando el audio del microfono, el programa:

1. Aplica una ventana de Hann a la señal para reducir la fuga espectral
2. Calcula la transformada de Fourier para obtener el espectro de frecuencias
3. Detecta los picos dominantes usando prominencia mínima configurable
4. Convierte cada frecuencia a su nota musical más cercana, incluyendo los cents de desafinación
5. Guarda los resultados en un CSV con su timestamp
6. Genera visualizaciones: forma de onda, espectro FFT y espectrograma

## Dependencias
 
| Librería | 
|---|
| `numpy` |
| `scipy` | 
| `matplotlib` |
| `pandas` |
| `sounddevice` |

## Instalación
 
```bash
pip install numpy scipy matplotlib pandas sounddevice
```

## Uso

```bash
# usar el micrófono por defecto
python escaner-frecuencias.py
 
# ver qué micrófonos están disponibles
python escaner-frecuencias.py --list-devices
 
# usar un micrófono específico (usar el número del comando anterior)
python escaner-frecuencias.py --device 2
 
# grabar durante x segundos y parar
python escaner-frecuencias.py --duration (cantidad de segundos)
```
