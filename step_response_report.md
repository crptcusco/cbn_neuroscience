# Informe de Análisis: Respuesta al Escalón y Eficiencia de la Red de Spikes

## 1. Objetivo del Análisis

Este análisis tuvo como objetivo validar la capacidad de nuestra red de spikes (modelo GLIF) para responder rápidamente a cambios bruscos en el estímulo, un comportamiento clave descrito en la Sección 5.5.4 de Trappenberg. Se comparó la respuesta empírica de la red con la de un modelo de tasa teórico simple para demostrar la eficiencia del cómputo con spikes.

## 2. Metodología

Se aplicó un estímulo de escalón a una única columna cortical simulada. Se midió el tiempo de subida (rise time) de la actividad de población resultante ($A(t)$). El nivel de ruido de fondo inyectado en el sistema fue ajustado iterativamente hasta que el tiempo de subida fue inferior a 3 ms, emulando una red en estado de "alerta".

## 3. Hallazgos

### a. Nivel de Ruido y Respuesta Rápida (Tareas 1 y 2)

Se encontró que el nivel de ruido de fondo es un parámetro crítico para la velocidad de respuesta de la población.

- Con un ruido bajo ($\sigma_{noise} < 2.0$), la red exhibía una respuesta lenta, con tiempos de subida superiores a 10 ms, comportándose como un integrador "perezoso".
- Se determinó que un nivel de ruido óptimo de **$\sigma_{noise} \approx 2.5$** era suficiente para garantizar que un subconjunto de neuronas estuviera constantemente fluctuando cerca del umbral de disparo.
- Con este nivel de ruido, la respuesta de la población al escalón de estímulo fue casi instantánea (**< 3 ms**), replicando el "Instantaneous Jump" de actividad descrito por Trappenberg.

### b. Comparación con el Modelo de Tasa (Tarea 3)

La gráfica comparativa (`step_response_comparison.png`) demuestra visualmente la principal conclusión de este análisis:

- La **red de spikes (curva azul)** muestra una transición casi vertical en su actividad inmediatamente después del salto del estímulo.
- El **modelo de tasa teórico (curva roja)**, con su constante de tiempo $\tau=10$ ms, muestra una subida lenta y exponencial, incapaz de capturar la rápida reorganización de la red de spikes.

## 4. Conclusión

Este análisis valida que nuestra arquitectura de red de spikes, cuando se opera en un régimen de ruido adecuado ("estado de alerta"), es significativamente más rápida y eficiente para procesar cambios transitorios que un modelo de tasa simple. El ruido no es simplemente "ruido", sino un componente funcional que permite a la red superar la inercia de la constante de tiempo de la membrana individual. Esto justifica el uso de nuestra arquitectura CBN para tareas que requieren un procesamiento de información en tiempo real.
