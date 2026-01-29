# Justificación de la Normalización del Acoplamiento (Eq. 5.52)

La Tarea 2 solicita verificar que el acoplamiento entre columnas siga la regla de escalado $w_{ij} = w_0 / N$, donde $N$ es el número de neuronas en la población. Este principio es fundamental para asegurar que la dinámica del modelo de campo medio sea escalable y no dependa del tamaño de la población subyacente.

## Verificación en Nuestro Modelo

Nuestro modelo de acoplamiento, tanto en la versión de spikes (GLIF) como en la de tasa (Mean Field), es consistente con este principio.

### 1. Modelo de Spikes (GLIF)

La entrada de acoplamiento de una capa presináptica a una post-sináptica se calculaba como:
`input = g_axial * np.mean(presynaptic_layer.spikes)`

Aquí:
- `presynaptic_layer.spikes` es un vector booleano de tamaño $N$.
- `np.mean(...)` calcula la fracción de neuronas que dispararon ($k/N$).
- `g_axial` actúa como el peso base, $w_0$.

La entrada total es, por tanto, $w_0 \cdot (k/N)$. Esta formulación ya está normalizada por $N$. Si el número de neuronas $N$ se duplica, pero la actividad fraccional se mantiene, la entrada total a la capa siguiente no cambia, cumpliendo el requisito de escalabilidad.

### 2. Modelo de Tasa (Mean Field)

La entrada de acoplamiento se calcula ahora como:
`input = g_axial * presynaptic_layer.A`

Aquí:
- `presynaptic_layer.A` es la actividad de la población, que por definición es la tasa de disparo promedio del grupo de neuronas. Es el análogo directo de campo medio de la tasa de disparo fraccional $k/N$.
- `g_axial` sigue siendo el peso base, $w_0$.

La entrada es $w_0 \cdot A$. De nuevo, la normalización por $N$ es implícita en la propia definición de la actividad de población $A$.

**Conclusión:** La lógica de acoplamiento implementada es inherentemente escalable y consistente con el principio de $w_0/N$ descrito por Trappenberg en la Ecuación 5.52.
