# Bloque 2: La Computadora Neuronal (Acoplamiento No Lineal)

Este bloque demuestra cómo la arquitectura de red, a través de diferentes mecanismos de acoplamiento, puede realizar operaciones lógicas no lineales que son fundamentales para la computación. Los conceptos de lógica Sigma-Pi (AND), inhibición por shunting (control de ganancia) y compuertas dependientes de voltaje (NMDA) se ilustran a través de un experimento central de acoplamiento multiplicativo.

## Figura 4: Lógica Sigma-Pi (AND)

La red implementa una compuerta AND, donde la actividad de una columna de "compuerta" modula la transmisión de una columna de "señal". La salida solo es alta cuando ambas entradas están activas.

![Lógica AND](fig4_logica_and.png)

## Figura 5: Control de Ganancia (Inhibición por Shunting)

Este mismo mecanismo multiplicativo puede interpretarse como un control de ganancia, donde la columna de compuerta actúa como una inhibición por shunting que reduce drásticamente la ganancia de la columna de salida.

![Control de Ganancia](fig5_control_ganancia.png)

## Figura 6: Compuerta Dependiente de Contexto (NMDA)

El acoplamiento no lineal sirve como un análogo funcional de las sinapsis NMDA, donde una entrada (la compuerta) proporciona el contexto despolarizante necesario para que otra entrada (la señal) sea efectiva.

![Compuerta NMDA](fig6_compuerta_nmda.png)
