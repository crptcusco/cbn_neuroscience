# Informe Técnico: Análisis de Dinámica de Población

## 1. Determinación del τ Efectivo de la Población (Tarea 1)

Se simuló la respuesta de una columna cortical a un pulso de estímulo para determinar la constante de tiempo efectiva de la población ($\tau_{eff}$). La actividad de población empírica $A(t)$ fue extraída y se ajustó un modelo de decaimiento exponencial ($A(t) = A_0 e^{-t / \tau}$) a su fase de decaimiento.

- **Resultado:** El $\tau$ efectivo ajustado fue de **~90-120 ms** (el valor exacto varía ligeramente con cada ejecución debido al ruido, ej. 92.54 ms, 117.05 ms).

- **Análisis:** Este valor es significativamente mayor que la constante de tiempo de la membrana de las neuronas individuales ($\tau_m = 15$ ms). Este hallazgo es consistente con la teoría de Trappenberg, que sugiere que las interacciones recurrentes y la dinámica sináptica dentro de la población neuronal dan lugar a una inercia temporal o "memoria" a nivel de población que es mucho más lenta que la de sus componentes individuales. Nuestra red cumple con esta aproximación de manera cualitativa, aunque el valor cuantitativo es mayor de lo esperado, indicando una fuerte recurrencia efectiva.

## 2. Test de Estado Estacionario (Tarea 2)

Se aplicó un estímulo constante de larga duración a una columna aislada para verificar si su actividad alcanza un estado estacionario, como predice la Ecuación 5.50 ($A = g(I_{ext})$).

- **Resultado:** La actividad de población $A(t)$ convergió a un valor medio estable con una desviación estándar muy baja (ej. Media = 0.0123, Std = 0.0007).

- **Análisis:** La red prefiere un **estado estacionario estable** cuando está aislada. Esto confirma que la columna individual se comporta como un Leaky Integrator de Población. Este resultado es crítico, ya que implica que cualquier dinámica oscilatoria persistente (como las oscilaciones Gamma) en una red de múltiples columnas debe ser una **propiedad emergente del acoplamiento entre ellas**, y no una propiedad intrínseca de las columnas mismas.

## 3. Caracterización de la Función de Activación g(x) (Tarea 3)

Se caracterizó la función de activación de la población $g(x)$ midiendo la actividad estacionaria $A$ para un rango de intensidades de estímulo de entrada $I_{ext}$.

- **Resultado:** La gráfica de $I_{ext}$ vs. $A$ (ver `activation_function_g_x.png`) muestra una curva que es inicialmente creciente y luego se satura a un nivel de actividad máximo.

- **Análisis:** La forma de la curva es **sigmoidal**, no lineal ni de umbral brusco. Crece a medida que más neuronas son reclutadas por el estímulo, pero se satura debido al periodo refractario y a la dinámica inhibitoria efectiva (el término de fuga y el potencial de reversión). Este comportamiento es altamente consistente con las curvas de entrada-salida de poblaciones neuronales ruidosas descritas por Trappenberg (Figura 5.8).
