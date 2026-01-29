# Informe de Hallazgos: Acoplamiento No Lineal

## 1. Objetivo del Análisis

Este informe resume los resultados de la implementación de tres mecanismos de acoplamiento no lineal en la red neuronal, basados en el Capítulo 5.6 de Trappenberg: lógica multiplicativa (Sigma-Pi), inhibición divisiva (shunting) y sinapsis dependientes de voltaje (NMDA).

## 2. Metodología

Para soportar estas interacciones complejas, la arquitectura del simulador fue refactorizada. La clase `CompartmentalColumn` ahora es impulsada por un motor de reglas flexible, permitiendo definir diferentes tipos de conexiones (`additive`, `multiplicative`, `divisive`, `nmda`) entre capas y columnas. Se crearon simulaciones específicas para validar cada uno de estos mecanismos.

## 3. Hallazgos

### a. Lógica Multiplicativa (Sigma-Pi / AND)

- **Implementación:** Se definió una regla donde la entrada a una capa (`L5` en Col 0) era el producto de la actividad de dos fuentes (`L2/3` en Col 0 y `L5` en Col 1).
- **Resultado:** La simulación (`nonlinear_coupling_demo.png`) demostró con éxito el comportamiento de una compuerta AND. La capa objetivo solo se activó cuando ambas fuentes estaban activas simultáneamente, permaneciendo inactiva si solo una de ellas recibía estímulo.
- **Análisis:** Este mecanismo permite a la red realizar cómputos lógicos complejos que van más allá de la simple suma o el OR, siendo fundamental para tareas como la detección de coincidencias o el binding de características.

### b. Inhibición Divisiva (Shunting Inhibition)

- **Implementación:** Se creó una regla donde la actividad de una capa (`L6`) dividía la entrada total de otra capa (`L4`).
- **Resultado:** La simulación (`nonlinear_coupling_demo.png`) mostró que, ante un estímulo externo constante y fuerte en L4, la activación de la capa inhibitoria L6 provocó una reducción (división) de la actividad estacionaria de L4.
- **Análisis:** Este mecanismo actúa como un control de ganancia automático. En lugar de simplemente restar actividad, la modula en función del nivel de excitación, previniendo la saturación y la "excitación descontrolada" (runaway excitation). Es un mecanismo clave para la estabilización de la dinámica de la red.

### c. Sinapsis Dependientes de Voltaje (NMDA)

- **Implementación:** El modelo de neurona `LIF_NodeGroup` fue modificado para que las sinapsis de tipo `nmda` solo contribuyan a la corriente de entrada si el potencial de membrana de la neurona post-sináptica está por encima de un umbral (-60 mV).
- **Resultado:** La simulación (`nmda_gate_demo.png`) validó este comportamiento. Los spikes de una columna presináptica no tuvieron efecto en una columna post-sináptica en reposo. Sin embargo, cuando la columna post-sináptica fue "preparada" con una pequeña entrada despolarizante, los mismos spikes presinápticos provocaron una fuerte respuesta.
- **Análisis:** Las sinapsis NMDA actúan como detectores de coincidencia temporal y espacial. Permiten que las columnas "preparen" o "ceben" a sus vecinas, facilitando el disparo sincronizado y la formación de ensambles neuronales. Este mecanismo es crucial para el aprendizaje y la plasticidad sináptica.

## 4. Conclusión

La implementación exitosa de estas tres formas de acoplamiento no lineal eleva significativamente las capacidades computacionales del simulador. La red ya no está limitada a una simple lógica aditiva, sino que puede realizar operaciones lógicas, controlar su propia ganancia y formar asociaciones dinámicas basadas en el estado de la red.
