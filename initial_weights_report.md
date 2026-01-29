# Informe: Arquitectura para Plasticidad Funcional

## 1. Nueva Arquitectura

Para permitir que la red sea adaptativa, se ha refactorizado la arquitectura de simulación en tres componentes principales, desacoplando la dinámica neuronal, la conectividad y la orquestación de la simulación:

1.  **`CompartmentalColumn` (Núcleo Neuronal):**
    -   Su única responsabilidad es mantener el estado de sus capas (`LIF_NodeGroup`, `RateNodeGroup`, etc.).
    -   Es agnóstica a la red. Recibe un diccionario de inputs pre-calculados y actualiza sus capas.

2.  **`ConnectionManager` (Gestor de Conectividad):**
    -   Contiene la **matriz de pesos dinámicos (`W`)** de toda la red.
    -   Mantiene la lista de reglas de acoplamiento, que definen la estructura de la red (qué se conecta con qué y de qué forma).
    -   Proporciona métodos para leer, escribir y registrar la evolución de los pesos en la matriz `W`.

3.  **`NetworkSimulator` (Orquestador):**
    -   Es el motor principal de la simulación.
    -   Posee el `ConnectionManager` y el estado de todas las columnas.
    -   En cada paso de tiempo, utiliza el `ConnectionManager` para calcular todos los inputs de acoplamiento basándose en el estado actual de la red y los pesos dinámicos.
    -   Contiene el método `apply_plasticity(rule_func)`, que implementa el **framework de aprendizaje**. Este método permite que una regla de aprendizaje externa modifique los pesos en el `ConnectionManager` basándose en la actividad de la red.

## 2. Distribución de Pesos Iniciales

El `ConnectionManager` inicializa la matriz de pesos `W` basándose en el parámetro `weight` proporcionado en cada regla de la lista `coupling_rules`.

**Ejemplo:**
Una red con 4 capas (2 columnas con 2 capas cada una) y una regla que conecta la capa 0 con la 2 con un peso de `0.8` tendría una matriz de pesos inicial como la siguiente:

```
[[0. , 0. , 0. , 0. ],
 [0. , 0. , 0. , 0. ],
 [0.8, 0. , 0. , 0. ],  <-- Peso de la conexión
 [0. , 0. , 0. , 0. ]]
```

Todos los demás pesos son `0` por defecto. Esta matriz representa la conectividad inicial de la red. El `weight_history` (ver `weight_evolution.png`) demuestra cómo estos valores pueden cambiar dinámicamente durante la simulación gracias al nuevo framework de plasticidad.
