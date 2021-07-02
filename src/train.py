import random
from time import sleep, time

import gym

from model import QNN
from policy import EpsilonGreedyPolicy

if __name__ == "__main__":
    # パラメータ設定
    action_list = [-1, 1]  # 行動のリスト
    epsilon = 0.1  # ε-greedy法のパラメータ
    memory_size = 1000
    batch_size = 32
    dim_state = 3  # env.env.observation_space.shape[0]
    max_episode = 10
    max_step = 100
    warmup_steps = 1000

    # 環境作成
    env = gym.make("Pendulum-v0")
    qnn = QNN(dim_state, action_list)  # Double Network
    policy = EpsilonGreedyPolicy(qnn, epsilon)
    memory = list()  # Experience replayで使用するメモリ

    sleep(3)

    # Warm up: 訓練の前に経験をメモリに貯める
    # ランダムに行動し、経験を貯める
    step = 0
    total_step = 0
    state = env.reset()
    print("===Warm up===")
    while True:
        step += 1
        total_step += 1
        print(f"\r---Step{total_step:04d}---", end="")

        action = random.choice(action_list)
        epsilon, q_values = 1.0, None
        next_state, reward, done, info = env.step([action])
        # reward clipping
        if reward < -1:
            c_reward = -1
        else:
            c_reward = 1
        # 経験を貯める
        memory.append((state, action, c_reward, next_state, done))

        state = next_state
        if step > max_step:
            state = env.reset()
            step = 0
        if total_step > warmup_steps:
            break
    memory = memory[-memory_size:]

    sleep(3)

    # 訓練
    print("\n===Train===")
    for episode in range(max_episode):
        print(f"===Episode{episode:05d}===")
        state = env.reset()
        for step in range(max_step):
            start_time = time()
            # 行動を選択
            action, epsilon, q_values = policy.get_action(state, action_list)
            # 状態と報酬を取得
            next_state, reward, done, info = env.step([action])
            # reward clipping
            if reward < -1:
                c_reward = -1
            else:
                c_reward = 1
            # experience replay
            memory.append((state, action, c_reward, next_state, done))
            exps = random.sample(memory, batch_size)
            # メインネットワークのパラメータを更新
            loss, td_error = qnn.update_on_batch(exps)
            # ターゲットネットワークの重みの一部を同期
            qnn.sync_target_network(0.01)
            state = next_state
            memory = memory[-memory_size:]
            print(
                f"{time() - start_time:.1f} Step{step:05d} State: {state} Action: {action} -> R: {c_reward} State: {next_state}"
            )

    env.close()
