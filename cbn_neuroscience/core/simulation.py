# cbn_neuroscience/core/simulation.py

import numpy as np
import numba
from cbnetwork.localnetwork import LocalNetwork

TIME_STEP_MS = 10  # Cada paso de simulación representa 10ms

@numba.njit
def _find_external_pos(external_map: np.ndarray, var_index: int) -> int:
    for i in range(len(external_map)):
        if external_map[i] == var_index:
            return i
    return -1

@numba.njit
def _evaluate_clause(clause: np.ndarray, state: np.ndarray, external: np.ndarray, external_map: np.ndarray) -> bool:
    for literal in clause:
        is_negated = literal < 0
        var_index = abs(literal)

        if var_index < len(state) + 1:
            val = state[var_index - 1]
        else:
            ext_pos = _find_external_pos(external_map, var_index)
            val = external[ext_pos] if ext_pos != -1 else 0

        if is_negated:
            val = 1 - val

        if val == 1:
            return True
    return False

@numba.njit
def _accelerated_next_state(
    current_state: np.ndarray,
    rules: numba.typed.List,
    external_values_arr: np.ndarray,
    external_map: np.ndarray
) -> np.ndarray:
    next_state = np.zeros_like(current_state)
    num_vars = len(current_state)

    for i in range(num_vars):
        var_rules = rules[i]
        func_result = True
        for clause in var_rules:
            if not _evaluate_clause(clause, current_state, external_values_arr, external_map):
                func_result = False
                break
        next_state[i] = 1 if func_result else 0
    return next_state

class Simulator:
    def __init__(self, network: LocalNetwork):
        self.network = network
        self.variables_map = {
            var.index: var for var in self.network.descriptive_function_variables
        }
        self.simulation_time_ms = 0 # Seguimiento del tiempo de simulación

        self.is_dynamic = any(callable(var.cnf_function) for var in self.variables_map.values())

        if not self.is_dynamic:
            temp_rules = []
            sorted_indices = sorted(self.variables_map.keys())
            for var_index in sorted_indices:
                variable = self.variables_map[var_index]
                clauses_list = numba.typed.List()
                if variable.cnf_function:
                    for clause in variable.cnf_function:
                        clauses_list.append(np.array(clause, dtype=np.int32))
                temp_rules.append(clauses_list)
            self.static_rules = numba.typed.List(temp_rules)

    def _get_next_state_python(self, current_state: np.ndarray, external_values: dict = None) -> np.ndarray:
        if external_values is None:
            external_values = {}
        next_state = np.zeros_like(current_state)
        current_state_dict = {i + 1: current_state[i] for i in range(len(current_state))}

        for var_index in sorted(self.variables_map.keys()):
            variable = self.variables_map[var_index]
            cnf_function = variable.cnf_function

            if callable(cnf_function):
                cnf_function = cnf_function(external_values)

            next_val = LocalNetwork.evaluate_boolean_function(
                cnf_function, current_state_dict, external_values
            )
            next_state[var_index - 1] = next_val
        return next_state

    def run(self, initial_state: np.ndarray, steps: int, external_inputs: list = None) -> tuple[np.ndarray, str | None]:
        if len(initial_state) != len(self.network.internal_variables):
            raise ValueError("El tamaño del estado inicial no coincide con el número de variables de la red.")
        if external_inputs and len(external_inputs) != steps:
            raise ValueError("La longitud de external_inputs debe ser igual al número de pasos.")

        history = np.zeros((steps + 1, len(initial_state)), dtype=int)
        history[0] = initial_state
        current_state = initial_state.copy()
        external_map = np.array(sorted(self.network.external_variables), dtype=np.int32)

        for i in range(steps):
            current_external_dict = external_inputs[i] if external_inputs else {}

            if self.is_dynamic:
                current_state = self._get_next_state_python(current_state, current_external_dict)
            else:
                external_arr = np.array([current_external_dict.get(k, 0) for k in sorted(self.network.external_variables)], dtype=np.int32)
                current_state = _accelerated_next_state(current_state, self.static_rules, external_arr, external_map)

            history[i + 1] = current_state
            self.simulation_time_ms += TIME_STEP_MS # Incrementar el tiempo de simulación

        return history, None
