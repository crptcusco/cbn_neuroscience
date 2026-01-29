# Estructura de Resultados de la Tesis (Capítulos 5 y 6)

Este documento organiza los resultados experimentales clave del modelo de red neuronal computacional, divididos en tres bloques conceptuales. Cada bloque corresponde a una fase de validación del modelo, desde sus propiedades dinámicas fundamentales hasta su capacidad para el aprendizaje y la computación.

---

## Índice de Experimentos y Conclusiones

| Bloque | Experimento | Script de Prueba | Conclusión Principal |
| :--- | :--- | :--- | :--- |
| **1: Motor Biológico** | Constante de Tiempo Efectiva | `spike_vs_rate_model.py` | La dinámica de la red de picos puede ser descrita por un modelo de tasa con una constante de tiempo emergente. |
| | Función de Activación | `spike_vs_rate_model.py` | La población neuronal exhibe una función de activación no lineal sigmoidal que define su régimen de respuesta. |
| | Respuesta Rápida al Escalón | `step_response_analysis.py` | La red responde a cambios de estímulo en milisegundos, validando su capacidad para el procesamiento rápido. |
| **2: Computadora Neuronal** | Lógica Sigma-Pi (AND) | `nonlinear_coupling_demo.py` | El acoplamiento no lineal permite a la red realizar operaciones lógicas fundamentales como la multiplicación (AND). |
| | Control de Ganancia | `nonlinear_coupling_demo.py` | La inhibición por shunting se implementa como un control de ganancia multiplicativo sobre la actividad de una población. |
| | Compuerta de Contexto (NMDA) | `nonlinear_coupling_demo.py` | Las interacciones no lineales actúan como compuertas dependientes de contexto, análogas a las sinapsis NMDA. |
| **3: Aprendizaje y Estabilidad** | Estabilidad Gamma | `synaptic_competition_demo.py` | Los mecanismos homeostáticos aseguran que la red mantenga una actividad oscilatoria estable sin saturación. |
| | Poda Sináptica (Oja/LTD) | `synaptic_competition_demo.py` | La plasticidad competitiva refina la conectividad de la red, eliminando sinapsis débiles y fortaleciendo las eficaces. |
| | Aprendizaje Asociativo | `associative_learning_demo.py`| La red es capaz de aprendizaje Hebbiano, formando asociaciones entre patrones de estímulo concurrentes. |

---

### Contenido de los Bloques

*   **[Bloque 1: Dinámica de Población](./01_dinamica_poblacion.md)**
*   **[Bloque 2: Arquitectura No Lineal](./02_arquitectura_no_lineal.md)**
*   **[Bloque 3: Aprendizaje y Estabilidad](./03_aprendizaje_y_estabilidad.md)**
