from __future__ import print_function
import numpy as np
import copy


class amoeba():
    def __init__(self, environment_size=(8, 8), foods=1, obstacles=1, poisons=1, demo=False):
        def place_object():
            while True:
                row = np.random.randint(0, environment_size[0])
                col = np.random.randint(0, environment_size[1])
                if [row, col] not in self.foods + self.obstacles + self.poisons:
                    break
            return [row, col]

        def place_food():
            row = np.random.randint(0, environment_size[0])
            col = np.random.randint(0, environment_size[1])
            if self.environ[row][col] == 1:
                place_food()
            else:
                self.environ[row][col] = 1

        def place_amoeba():
            row = np.random.randint(0, environment_size[0])
            col = np.random.randint(0, environment_size[1])
            if self.environ[row][col] == 1:
                return place_amoeba()
            else:
                return [row, col]

        def one_d_project(pt):
            return pt[0] * self.environ.shape[1] + pt[1]

        def build_transition_matrix():
            def render_next_state(r, c, a):
                # 0 = up, 1 = left, 2 = down, 3 = right
                s_prime = np.zeros(self.len)
                if [r, c] in self.foods + self.obstacles + self.poisons:
                    return s_prime
                if a == 0:
                    if r - 1 >= 0 and [r - 1, c] not in self.obstacles:
                        s_prime[one_d_project((r - 1, c))] += 0.8
                    else:
                        s_prime[one_d_project((r, c))] += 0.8
                    if c + 1 < self.c and [r, c + 1] not in self.obstacles:
                        s_prime[one_d_project((r, c + 1))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                    if c - 1 >= 0 and [r, c - 1] not in self.obstacles:
                        s_prime[one_d_project((r, c - 1))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                elif a == 1:
                    if c - 1 >= 0 and [r, c - 1] not in self.obstacles:
                        s_prime[one_d_project((r, c - 1))] += 0.8
                    else:
                        s_prime[one_d_project((r, c))] += 0.8
                    if r + 1 < self.r and [r + 1, c] not in self.obstacles:
                        s_prime[one_d_project((r + 1, c))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                    if r - 1 >= 0 and [r - 1, c] not in self.obstacles:
                        s_prime[one_d_project((r - 1, c))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                elif a == 2:
                    if r + 1 < self.r and [r + 1, c] not in self.obstacles:
                        s_prime[one_d_project((r + 1, c))] += 0.8
                    else:
                        s_prime[one_d_project((r, c))] += 0.8
                    if c + 1 < self.c and [r, c + 1] not in self.obstacles:
                        s_prime[one_d_project((r, c + 1))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                    if c - 1 >= 0 and [r, c - 1] not in self.obstacles:
                        s_prime[one_d_project((r, c - 1))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                elif a == 3:
                    if c + 1 < self.c and [r, c + 1] not in self.obstacles:
                        s_prime[one_d_project((r, c + 1))] += 0.8
                    else:
                        s_prime[one_d_project((r, c))] += 0.8
                    if r + 1 < self.r and [r + 1, c] not in self.obstacles:
                        s_prime[one_d_project((r + 1, c))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                    if r - 1 >= 0 and [r - 1, c] not in self.obstacles:
                        s_prime[one_d_project((r - 1, c))] += 0.1
                    else:
                        s_prime[one_d_project((r, c))] += 0.1
                return s_prime

            T = np.zeros((self.len, self.len, 4))

            one_d_pt = 0
            for row in range(self.r):
                for col in range(self.c):
                    for act in range(0, 4):
                        T[one_d_pt, :, act] = render_next_state(row, col, act)
                    one_d_pt += 1
            return T

        def build_reward_matrix(normal_state_value=-0.04):
            env_vec = np.array([normal_state_value for i in range(self.len)])
            for food in self.foods:
                env_vec[one_d_project(food)] = 1.0
            for poison in self.poisons:
                env_vec[one_d_project(poison)] = -1.0
            for obstacle in self.obstacles:
                env_vec[one_d_project(obstacle)] = .0
            return env_vec

        self.environ = np.zeros(environment_size)
        self.len = environment_size[0] * environment_size[1]
        self.r, self.c = environment_size[0], environment_size[1]
        self.vector = np.zeros((1, self.len))
        if demo == True:
            self.obstacles = [[1, 1]]
            self.foods = [[0, 3]]
            self.poisons = [[1, 3]]
            self.location = [2, 0]
        else:
            self.foods = []
            self.obstacles = []
            self.poisons = []
            self.location = place_object()
            for f in range(foods):
                self.foods.append(place_object())
            for o in range(obstacles):
                self.obstacles.append(place_object())
            for p in range(poisons):
                pt = place_object()
                self.poisons.append(pt)
        self.t = build_transition_matrix()
        print(self.environ)
        self.reward_matrix = build_reward_matrix()

        print(self.t[:, :, 0])

    def return_transition_matrix(self):
        return self.t

    def return_value_matrix(self, state, utility_vec, reward, gamma):
        # array to store that utility generated for each possible action given the state
        action_utility_array = np.zeros(4)
        for act in range(4):
            # dot the state with the transition matrix given the action
            # basic if we have this state, and this action what are the probabilities
            # that our amoeba will end up in any of the next states in the environment
            # the transtition matrix already has this info in it
            # we are just filtering down the transition matrix
            # using this dot product operation
            # print("state:\n",state)
            # print("t matrix:\n",self.t[:,:,act])
            lookup_result = np.dot(state, self.t[:, :, act])
            # print("lookup:\n", lookup_result)

            # once we have model of the outcome of an action given a state
            # then rate the value of those outcomes probabilities
            # by bouncing the utility vector off the outcome probs
            # this utility vector is something we are iteratively updating
            # in the value iteration algorithm

            # print("utility:\n", utility_vec)
            action_utility = np.sum(np.multiply(utility_vec, lookup_result))
            # bounce to lookup results aka the position where you up probabilities
            # off the vector that holds the utilities for positions in that environment
            action_utility_array[act] = action_utility

        return reward + gamma * np.max(action_utility_array)

    def value_iter_env(self, n, gamma=0.999, epsilon=0.01):
        c = 0
        g_list = []

        # utility vectors
        u = np.zeros(self.len)
        u_prime = np.zeros(self.len)

        # value iteration loop
        while True:
            delta = 0
            u = u_prime.copy()
            c += 1
            g_list.append(u)
            for state in range(self.len):
                reward = self.reward_matrix[state]
                v = np.zeros((1, self.len))
                v[0, state] = 1.0
                u_prime[state] = self.return_value_matrix(v, u, reward, gamma)
                delta = max(delta, np.abs(u_prime[state] - u[state]))  # stopping criteria
            if delta < epsilon * (1 - gamma) / gamma:
                print("Iterations: {}".format(n))
                print("Delta: {}".format(delta))
                print("Gamma: {}".format(gamma))
                print("Epsilon: {}".format(epsilon))
                print(u.reshape(self.environ.shape))
                self.policy = u
                break

    def print_environ(self):
        border = ''.join(["-" for i in range((4 * self.environ.shape[1]) + 1)])
        print(border)
        env = copy.deepcopy(self.environ)
        env[self.location[0]][self.location[1]] = 1
        for obj in self.foods:
            env[obj[0], obj[1]] = 3
        for obj in self.obstacles:
            env[obj[0], obj[1]] = 5
        for obj in self.poisons:
            env[obj[0], obj[1]] = -1
        for row in env:
            row_string = ['|']
            for col in row:
                if col == 1:
                    row_string += [' @ |']
                elif col == 3:
                    row_string += [' $ |']
                elif col == -1:
                    row_string += [' X |']
                elif col == 5:
                    row_string += ['||||']
                else:
                    row_string += ['   |']
            print(''.join(row_string))
            print(border)

    def move_south(self):
        limits = self.environ.shape
        if self.location[0] + 1 != limits[0]:
            self.location[0] += 1

    def move_north(self):
        limits = self.environ.shape
        if self.location[0] - 1 >= 0:
            self.location[0] -= 1

    def move_east(self):
        limits = self.environ.shape
        if self.location[1] + 1 != limits[1]:
            self.location[1] += 1

    def move_west(self):
        limits = self.environ.shape
        if self.location[1] - 1 >= 0:
            self.location[1] -= 1


if __name__ == "__main__":
    # np.random.seed(2111)
    # amoeba = amoeba()
    # amoeba(environment_size=(3,4), foods=1, obstacles=1, poisons=1).print_environ()
    a = amoeba(environment_size=(3, 4), demo=True)
    # a = amoeba(environment_size=(3,4))
    # a = amoeba(environment_size=(10,10))
    a.print_environ()
    print(a.reward_matrix)
    print(a.reward_matrix.reshape(a.environ.shape))
    a.value_iter_env(50)
    a.value_iter_env(9, gamma=0.5, epsilon=0.001)