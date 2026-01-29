# examples/associative_learning_demo.py

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuración del Experimento (Tarea 2) ---

# Definir los patrones de entrada
# Usaremos un vector de 12 canales de entrada
N_INPUTS = 12
pattern_A_channels = [1, 3, 5]
pattern_B_channels = [8, 9, 10]

pattern_A = np.zeros(N_INPUTS)
pattern_A[pattern_A_channels] = 1.0

pattern_B = np.zeros(N_INPUTS)
pattern_B[pattern_B_channels] = 1.0

# Pesos sinápticos iniciales
weights = np.zeros(N_INPUTS)
weights[pattern_A_channels] = 1.0 # Conexiones fuertes para el patrón ancla
weights[pattern_B_channels] = 0.0 # Sin conexión para el patrón nuevo

# Parámetros del modelo
THRESHOLD = 1.5
LEARNING_RATE = 0.1

# --- 2. Funciones del Modelo y Test Inicial ---

def calculate_output(weights, input_pattern):
    """Calcula la activación 'h' y la salida 'r_out'."""
    h = np.dot(weights, input_pattern)
    r_out = 1.0 if h > THRESHOLD else 0.0
    return h, r_out

# --- Test Antes del Aprendizaje ---
print("--- Fase de Test (Antes del Aprendizaje) ---")
h_before, r_out_before = calculate_output(weights, pattern_B)
print(f"Respuesta al Patrón B (antes): h = {h_before:.2f}, r_out = {r_out_before}")
assert r_out_before == 0.0, "La neurona no debería disparar al Patrón B antes del aprendizaje."

print("\nProtocolo experimental implementado.")


# --- 3. Fase de Entrenamiento (Tarea 1) ---

print("\n--- Fase de Entrenamiento ---")
training_pattern = pattern_A + pattern_B
weights_history = [] # Para registrar la evolución de los pesos de B

# Copia de los pesos para el entrenamiento
trainable_weights = weights.copy()

for step in range(6):
    # Guardar los pesos actuales de los canales B
    weights_history.append(trainable_weights[pattern_B_channels].copy())

    # Calcular la salida con el patrón combinado
    h_train, r_out_train = calculate_output(trainable_weights, training_pattern)

    # Aplicar la regla de aprendizaje Hebbiano (Δw = η * r_in * r_out)
    if r_out_train > 0:
        delta_w = LEARNING_RATE * training_pattern * r_out_train
        trainable_weights += delta_w

    print(f"Paso {step+1}: h = {h_train:.2f}, r_out = {r_out_train:.1f}, Nuevos pesos B = {trainable_weights[pattern_B_channels]}")

# Guardar los pesos finales
final_weights = trainable_weights
weights_history.append(final_weights[pattern_B_channels].copy())
print("Entrenamiento completado.")


# --- 4. Fase de Test (Después del Aprendizaje) ---

print("\n--- Fase de Test (Después del Aprendizaje) ---")
h_after, r_out_after = calculate_output(final_weights, pattern_B)
print(f"Respuesta al Patrón B (después): h = {h_after:.2f}, r_out = {r_out_after}")
assert r_out_after == 1.0, "La neurona DEBERÍA disparar al Patrón B después del aprendizaje."


# --- 5. Visualización (Tarea 3) ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfica 1: Antes y Después del Aprendizaje
ax1.bar(['Antes', 'Después'], [h_before, h_after], color=['gray', 'lightblue'])
ax1.axhline(THRESHOLD, color='r', linestyle='--', label=f'Umbral θ = {THRESHOLD}')
ax1.set_title('Respuesta al Patrón B\n(Antes vs. Después del Aprendizaje)')
ax1.set_ylabel('Activación (h)')
ax1.legend()
ax1.grid(axis='y', linestyle=':')

# Gráfica 2: Evolución de los Pesos
weights_history = np.array(weights_history)
for i in range(weights_history.shape[1]):
    ax2.plot(weights_history[:, i], 'o-', label=f'Peso del Canal {pattern_B_channels[i]}')

ax2.set_title('Evolución de los Pesos Sinápticos del Patrón B')
ax2.set_xlabel('Paso de Entrenamiento')
ax2.set_ylabel('Fuerza del Peso (w)')
ax2.set_xticks(range(7))
ax2.set_xticklabels(['Inicial'] + [str(i+1) for i in range(6)])
ax2.legend()
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('associative_learning_demo.png')
print("\nGráfica de resultados guardada en 'associative_learning_demo.png'")
