# Informe de Resultados: Experimento de Aprendizaje Asociativo

## 1. Objetivo

El objetivo de este experimento fue demostrar la capacidad de una neurona simple para aprender a responder a un estímulo nuevo (Patrón B) después de haber sido presentado repetidamente en asociación con un estímulo conocido que ya provoca una respuesta (Patrón A). Este es un ejemplo canónico de aprendizaje Hebbiano, basado en la Figura 6.3 de Trappenberg.

## 2. Metodología

Se configuró una única neurona con un umbral de activación $\theta = 1.5$. La neurona recibía inputs de 12 canales.

- **Patrón A (Ancla):** Se activaron los canales [1, 3, 5]. Los pesos sinápticos iniciales para estos canales se fijaron en `w=1.0`.
- **Patrón B (Nuevo):** Se activaron los canales [8, 9, 10]. Los pesos iniciales para estos se fijaron en `w=0.0`.

El aprendizaje se implementó con una regla Hebbiana incremental: $\Delta w = \eta \cdot r_{in} \cdot r_{out}$, con $\eta = 0.1$.

## 3. Fases del Experimento y Resultados

### a. Fase de Test (Antes del Aprendizaje)

Se presentó únicamente el Patrón B a la neurona.
- **Resultado:** La activación total fue $h = \sum w_i \cdot r_i = 0.0 \cdot 1 + 0.0 \cdot 1 + 0.0 \cdot 1 = 0.0$.
- **Conclusión:** Como $h < \theta$, la neurona no disparó.

### b. Fase de Entrenamiento

Se presentaron ambos patrones (A y B) simultáneamente durante 6 pasos.
- **Resultado:** La activación en el primer paso fue $h = (1.0 \cdot 3) + (0.0 \cdot 3) = 3.0$. Como $h > \theta$, la neurona disparó ($r_{out}=1.0$).
- La regla de aprendizaje se activó, y los pesos de los canales del Patrón B se incrementaron en `0.1` en cada uno de los 6 pasos.
- **Conclusión:** Los pesos finales para los canales [8, 9, 10] fueron `w=0.6`. La evolución de estos pesos se puede ver en la gráfica `associative_learning_demo.png`.

### c. Fase de Test (Después del Aprendizaje)

Se presentó únicamente el Patrón B a la neurona con los pesos ya modificados.
- **Resultado:** La activación total fue $h = \sum w_i \cdot r_i = 0.6 \cdot 1 + 0.6 \cdot 1 + 0.6 \cdot 1 = 1.8$.
- **Conclusión:** Como $h > \theta$ ($1.8 > 1.5$), la neurona **disparó**.

## 4. Conclusión Final

El experimento fue un éxito. La neurona aprendió exitosamente la asociación. Al presentar repetidamente el Patrón B junto con el Patrón A (que causaba el disparo), la conexión sináptica del Patrón B se fortaleció hasta el punto en que el Patrón B por sí solo fue suficiente para provocar una respuesta. Esto valida la implementación de la regla de aprendizaje Hebbiano y demuestra la capacidad de la red para el aprendizaje asociativo.
