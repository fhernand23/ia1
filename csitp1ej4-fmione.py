from enum import Enum
import numpy as np

# número de iteraciones
N = 50


# posibles estados del agente (espacio de percepciones)
class State(Enum):
    BOTTOM = "BOTTOM"
    MIDDLE = "MIDDLE"
    TOP = "TOP"


# posibles acciones a efectuar por el agente (espacio de acciones)
class Action(Enum):
    REST = "REST"
    CLIMB = "CLIMB"
    DOWN = "DOWN"


# definición del entorno
class Environment:

    def __init__(self):
        # define el entorno (current state) => [(action) => (next state, reward)]
        self.definition = {
            State.BOTTOM: {
                Action.REST: (State.BOTTOM, 0.1),
                Action.CLIMB: (State.TOP, 0.0)
            },
            State.MIDDLE: {
                Action.CLIMB: (State.TOP, 0.3),
                Action.DOWN: (State.BOTTOM, 1.0)
            },
            State.TOP: {
                Action.REST: (State.TOP, 0.1),
                Action.DOWN: (State.MIDDLE, 0.2)
            }
        }

    # función que retorna las posibles acciones de un determinado estado
    def get_possible_action(self, state):
        return list(self.definition[state].keys())

    # función que retorna las posibles acciones de un determinado estado y la recompensa de cada una
    def get_possible_action_reward(self, state):
        return list(self.definition[state].items())

    # función que permite al agente moverse (realizar una accion según su estado actual)
    def execute_action(self, state, action):
        # controla que la accion se pueda ejecutar en el estado actual
        if action in list(self.definition[state].keys()):
            reward = self.definition[state][action][1]
            observation = self.definition[state][action][0]
            return [observation, reward]
        else:
            return None


# definición del agente
class Agent:

    def __init__(self):
        # estado inicial
        self.state = State.BOTTOM

        # historia de percepciones recibidas
        self.perception_history = []

        # factor de descuento (da mayor o menor peso a las recompensas pasadas)
        self.gamma = 1

        # muestra el estado actual
        print("Estado inicial: BOTTOM")
        print("recompensa acumulada: 0")

    def move(self, environment, action):
        perception = environment.execute_action(self.state, action)
        if perception is not None:
            observation, reward = perception
            self.state = observation
            self.perception_history.append([action, observation, reward])
            print("Acción: {}".format(action))
            print("Estado: {}".format(self.state))
            print("Recompensa acumulada: %.2f" % self.get_accumulated_reward())
        else:
            print("Movimiento no permitido - Existe un error")

    def get_accumulated_reward(self):
        sum_reward = 0
        for k, [_, _, reward] in enumerate(self.perception_history):
            sum_reward += (self.gamma ** k) * reward
        return sum_reward


# agente que toma la mejor recompensa instantanea
def greedy_agent():
    # crea el agente
    agent = Agent()
    # crea el entorno
    environment = Environment()

    for i in range(N):
        # obtiene las posibles acciones y sus recompensas
        actions = environment.get_possible_action_reward(agent.state)
        # revisa cual tiene mejor recompensa
        if actions[0][1][1] < actions[1][1][1]:
            agent.move(environment, actions[1][0])
        else:
            agent.move(environment, actions[0][0])


# agente que elige la proxima accion aleatoriamente (es igual a e-random con e=0.5)
def random_agent():
    # crea el agente
    agent = Agent()
    # crea el entorno
    environment = Environment()

    for i in range(N):
        # obtiene las posibles acciones y elige una aleatoriamente
        actions = environment.get_possible_action(agent.state)
        random_action = int(np.random.rand() * 2)
        agent.move(environment, actions[random_action])


# agente que elige la proxima accion que menos recompensa tiene con probabilidad e
def e_random_agent(e):
    # crea el agente
    agent = Agent()
    # crea el entorno
    environment = Environment()

    for i in range(N):
        # obtiene las posibles acciones y sus recompensas
        actions = environment.get_possible_action_reward(agent.state)

        min_action = 0 if actions[0][1][1] < actions[1][1][1] else 1

        # elige la accion con minima recompensa con una probabilidad e
        if np.random.rand() <= e:
            agent.move(environment, actions[min_action][0])
        else:
            agent.move(environment, actions[1 if min_action == 0 else 0][0])


# agente smart que varia el parametro 'e' segun la cantidad de visitas al estado
def smart_agent(e):
    # crea el agente
    agent = Agent()
    # crea el entorno
    environment = Environment()
    # contador de estados para variar probabilidad y agregar inteligencia
    state_counter = {}

    for i in range(N):
        # obtiene las posibles acciones y sus recompensas
        actions = environment.get_possible_action_reward(agent.state)

        min_action = 0 if actions[0][1][1] < actions[1][1][1] else 1

        # probabilidad de visitar el estado con menor recompenza inmediata (modificado segun visitas)
        e_variant = e

        # revisa si existe un contador del estado actual para actualizar la probabilidad e
        if agent.state.name in state_counter:
            state_counter[agent.state.name] += 1
            e_variant = e + (state_counter[agent.state.name] / N)
        else:
            state_counter[agent.state.name] = 1

        # elige la accion con minima recompensa con una probabilidad e
        if np.random.rand() <= e_variant:
            agent.move(environment, actions[min_action][0])
        else:
            agent.move(environment, actions[1 if min_action == 0 else 0][0])


# ejecuta agentes
random_agent()
greedy_agent()
e_random_agent(0.2)
smart_agent(0.2)