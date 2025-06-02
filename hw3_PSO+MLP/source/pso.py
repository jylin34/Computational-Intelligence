import numpy as np
from geometry import border_to_segments, is_circle_near_segment, distance_to_goal

class PSO:
    def __init__(self, particle_count, cognition_rate, social_rate, inertia_weight, mlp, car, goal_tl, goal_br, log_function=None):
        self.particle_count = particle_count
        self.cognition_rate = cognition_rate
        self.social_rate = social_rate
        self.inertia_weight = inertia_weight
        self.mlp = mlp
        self.car = car
        self.goal_tl = goal_tl
        self.goal_br = goal_br
        self.log_function = log_function

        # 初始化粒子位置與速度
        self.particles = [self.initialize_particle() for _ in range(particle_count)]
        self.velocities = [self.initialize_velocity() for _ in range(particle_count)]

        # 初始化最佳位置與全域最佳
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = [float('inf')] * particle_count
        self.global_best_position = self.particles[0].copy()
        self.global_best_score = float('inf')

    def initialize_particle(self):
        # 初始化粒子位置（隨機權重與偏置）
        return np.random.uniform(-1, 1, self.get_particle_size())

    def initialize_velocity(self):
        # 初始化粒子速度
        return np.random.uniform(-0.1, 0.1, self.get_particle_size())

    def get_particle_size(self):
        # 計算粒子維度（MLP 的所有權重與偏置展平後的大小）
        return (
            self.mlp.weights_input_hidden.size +
            self.mlp.bias_hidden.size +
            self.mlp.weights_hidden_output.size +
            self.mlp.bias_output.size
        )

    def decode_particle(self, particle):
        # 將粒子解碼為 MLP 的權重與偏置
        # 一個 particle 是 38 dimensional numpy.ndarray 向量
        # 這個function做的事情就是把 particle 分成四個部分，分別對應到 MLP 的權重與偏置
        input_hidden_size = self.mlp.weights_input_hidden.size
        hidden_size = self.mlp.bias_hidden.size
        hidden_output_size = self.mlp.weights_hidden_output.size

        weights_input_hidden = particle[:input_hidden_size].reshape(self.mlp.weights_input_hidden.shape)
        bias_hidden = particle[input_hidden_size:input_hidden_size + hidden_size]
        weights_hidden_output = particle[input_hidden_size + hidden_size:input_hidden_size + hidden_size + hidden_output_size].reshape(self.mlp.weights_hidden_output.shape)
        bias_output = particle[input_hidden_size + hidden_size + hidden_output_size:]

        return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    def fitness_function(self, border_points, steps):
        """
        計算適應度值 (fitness value)
        :param border_points: 邊界線段
        :return: fitness value, 是否結束 (True 表示撞牆或抵達終點)
        """
        # 每走一步加 0.1
        distance = distance_to_goal(self.car.x, self.car.y, self.goal_tl, self.goal_br)
        fitness = -steps + distance

        # 檢查是否抵達終點
        if distance <= 3:
            fitness -= 100  # 抵達終點減去 100
            print("Reached the goal!")
            return fitness, True

        # 檢查是否撞牆
        for x1, y1, x2, y2 in border_to_segments(border_points):
            if is_circle_near_segment(self.car.x, self.car.y, 3, x1, y1, x2, y2):  # 假設車輛半徑為 1 車輛半徑忘記是什麼了
                fitness += 100  # 撞牆加 100
                print("Hit the wall!")
                return fitness, True

        return fitness, False

    def evaluate_particle_step(self, steps, particle_index, border_points, step_callback=None):
        """
        執行粒子的單一步驟，更新車輛狀態並計算適應度
        :param particle_index: 當前粒子的索引
        :param border_points: 邊界線段
        :param step_callback: 每一步執行後的回調函數，用於更新動畫
        :return: 是否完成（True 表示撞牆或抵達終點）
        """
        particle = self.particles[particle_index]

        # 解碼粒子並更新 MLP 權重
        weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = self.decode_particle(particle)
        self.mlp.update_weights(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

        # 使用 MLP 決策車輛行動
        sensor_data = self.car.get_sensor_distances(border_to_segments(border_points))
        action_probabilities = self.mlp.forward(np.array(sensor_data))
        # print(action_probabilities)
        angles = np.array([-40, 0, 40])
        angle = np.sum(action_probabilities * angles)
        # print(angle)

        # 接著讓車輛以該角度前進
        self.car.move_forward(angle)

        # action = np.argmax(action_probabilities)

        # # 根據行動更新車輛位置
        # if action == 0:  # 左轉
        #     self.car.move_forward(-40)
        # elif action == 1:  # 直行
        #     self.car.move_forward(0)
        # elif action == 2:  # 右轉
        #     self.car.move_forward(40)
        

        # 調用回調函數更新動畫
        if step_callback:
            step_callback()

        # 記錄 log
        # if self.log_function:
            # self.log_function(f"Particle {particle_index + 1} action: {action}, position: ({self.car.x:.2f}, {self.car.y:.2f}), theta: {self.car.theta:.1f}°")
            # print(f"Particle {particle_index + 1} action: {action}, position: ({self.car.x:.2f}, {self.car.y:.2f}), theta: {self.car.theta:.1f}°")
        
        # 計算當前步驟的 fitness
        step_fitness, done = self.fitness_function(border_points, steps)

        if step_fitness < self.personal_best_scores[particle_index]:
            self.personal_best_positions[particle_index] = self.particles[particle_index].copy()
            self.personal_best_scores[particle_index] = step_fitness

        # self.personal_best_scores[particle_index] += step_fitnesss

        return done

    def optimize_step(self, border_points):
        """
        執行一次 PSO 優化步驟，更新粒子的位置與速度
        :param border_points: 邊界線段
        """
        for i, particle in enumerate(self.particles):
            # 更新個體最佳
            if self.personal_best_scores[i] < self.global_best_score:
                self.global_best_score = self.personal_best_scores[i]
                self.global_best_position = self.personal_best_positions[i]

        # 更新粒子速度與位置
        for i in range(self.particle_count):
            r1, r2 = np.random.rand(), np.random.rand()
            # r1 ,r2 = 1, 1
            # print(r1, r2)
            cognitive = self.cognition_rate * r1 * (self.personal_best_positions[i] - self.particles[i]) ## 有問題
            social = self.social_rate * r2 * (self.global_best_position - self.particles[i]) ## 有問題
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive + social
            self.particles[i] += self.velocities[i]

    def save_best_parameters(self, filename="best_parameters.txt"):
        """
        將最佳參數儲存到 .txt 檔案中
        :param filename: 儲存檔案的名稱
        """
        with open(filename, "w") as file:
            for param in self.global_best_position:
                file.write(f"{param}\n")
        if self.log_function:
            self.log_function(f"✅ Best parameters saved to {filename}")