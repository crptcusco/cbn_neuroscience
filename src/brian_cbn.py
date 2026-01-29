import sys
import os
import numpy as np
from brian2 import *

# --- EVITAR ERRORES DE COMPILACIÓN ---
prefs.codegen.target = 'numpy'

# --- RUTAS ---
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs', 'src', 'cbnetwork', 'src'))
sys.path.insert(0, lib_path)

from cbnetwork.cbnetwork import CBN
from cbnetwork.directededge import DirectedEdge
from cbnetwork.internalvariable import InternalVariable
from cbnetwork.localnetwork import LocalNetwork

def build_brian_to_cbn(neuron_group, synapses):
    l_local_networks = []
    l_directed_edges = []
    
    n_local_nets = len(neuron_group)
    n_var_per_net = 1 # Por ahora 1 variable por neurona para simplicidad
    n_total_internal = n_local_nets * n_var_per_net
    
    # 1. Crear Redes Locales
    for i in range(1, n_local_nets + 1):
        # IDs: 1, 2, 3...
        vars_ids = [int(i)]
        o_ln = LocalNetwork(i, vars_ids)
        l_local_networks.append(o_ln)

    # 2. Crear Directed Edges (Sinapsis)
    index_signal = 1
    index_var_signal = n_total_internal + 1
    
    for s_idx in range(len(synapses)):
        src = int(synapses.i[s_idx] + 1)
        tgt = int(synapses.j[s_idx] + 1)
        
        # En Brian, una sinapsis es simple. En tu CBN, usa variables de salida.
        l_out_vars = [src] 
        coupling_func = f" {src} " # Lógica simple de identidad
        
        o_edge = DirectedEdge(
            index=index_signal,
            index_variable_signal=index_var_signal,
            input_local_network=tgt,
            output_local_network=src,
            l_output_variables=l_out_vars,
            coupling_function=coupling_func
        )
        l_directed_edges.append(o_edge)
        index_var_signal += 1
        index_signal += 1

    # 3. Generar Dinámica (CNF)
    for o_ln in l_local_networks:
        # Buscar señales que entran a esta neurona/red
        input_sigs = [e for e in l_directed_edges if e.input_local_network == o_ln.index]
        o_ln.process_input_signals(input_sigs)
        
        # Forzar tipos int en total_variables para evitar ValueError
        o_ln.total_variables = [int(v) for v in o_ln.total_variables]
        
        for i_var in o_ln.internal_variables:
            # Lógica: v_next = (v_actual AND NOT signals) si es inhibitoria...
            # Para este ejemplo: v_next = OR de todas las entradas (Excitatorio)
            if not input_sigs:
                # Si no hay entrada, se mantiene estable
                cnf = [[int(i_var)]]
            else:
                # Cláusula disyuntiva: [Var1, Sig1, Sig2...]
                clause = [int(i_var)] + [int(s.index_variable) for s in input_sigs]
                cnf = [clause]
            
            o_var_model = InternalVariable(int(i_var), cnf)
            o_ln.descriptive_function_variables.append(o_var_model)

    return CBN(l_local_networks, l_directed_edges)

# --- ESCENARIO ---
G = NeuronGroup(3, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
S = Synapses(G, G, on_pre='v += 0.6')
S.connect(i=[0, 1, 2], j=[1, 2, 0])

o_cbn = build_brian_to_cbn(G, S)

print("INFO: Buscando atractores en la red traducida...")
o_cbn.find_local_attractors_sequential()
o_cbn.show_local_attractors()

# --- ANÁLISIS DE ESTABILIDAD GLOBAL ---
print("\n" + "="*50)
print("INICIANDO ANÁLISIS DE CAMPOS DE ATRACCIÓN")
print("="*50)

# 1. Encontrar pares compatibles (necesario para los campos)
o_cbn.find_compatible_pairs()

# 2. Encontrar y montar los campos de atracción
# Nota: Verifica si en tu versión es 'find_attractor_fields' o 'mount_stable_attractor_fields'
try:
    o_cbn.find_attractor_fields()
    print("Campos de atracción encontrados.")
except AttributeError:
    # Si la versión usa el nombre del script manual:
    o_cbn.mount_stable_attractor_fields()
    print("Campos de atracción montados.")

# 3. Mostrar resultados
o_cbn.show_attractor_fields()

# 4. Análisis de Escenas Globales (La estabilidad total del sistema)
o_cbn.generate_global_scenes()
o_cbn.show_global_scenes()

print("\nConteo final de campos por escena:")
print(o_cbn.count_fields_by_global_scenes())