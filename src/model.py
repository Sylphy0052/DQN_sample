import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dot, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def idx2mask(idx, max_size):
    mask = np.zeros(max_size)
    mask[idx] = 1.0
    return mask


class QNN:
    def __init__(self, dim_state: int, action_list, gamma=0.99, lr=1e-3, double_mode=True) -> None:
        self.dim_state = dim_state
        self.action_list = action_list
        self.action_len = len(action_list)
        self.optimizer = Adam(lr=lr)
        self.gamma = gamma
        self.double_mode = double_mode

        self.main_network = self.build_graph()
        self.target_network = self.build_graph()
        self.trainable_network = self.build_trainable_graph(self.main_network)

    def build_graph(self) -> Model:
        """build_graph ネットワークを作成する

        Returns:
            Model: NNモデル
        """
        nb_dense1 = self.dim_state * 10  # 30
        nb_dense3 = self.action_len * 10  # 20
        nb_dense2 = int(np.sqrt(nb_dense1 * nb_dense3))  # sqrt(600) → 24

        # Layer
        # Input: 入力層
        # Dense: 全結合層
        # 次元: 3→30→24→20→2
        l_input = Input(shape=(self.dim_state,), name="input_state")
        l_dense1 = Dense(nb_dense1, activation="relu", name="hidden1")(l_input)
        l_dense2 = Dense(nb_dense2, activation="relu", name="hidden2")(l_dense1)
        l_dense3 = Dense(nb_dense3, activation="relu", name="hidden3")(l_dense2)
        l_output = Dense(self.action_len, activation="linear", name="output")(l_dense3)
        model = Model(inputs=[l_input], outputs=[l_output])
        model.summary()
        # 訓練の設定
        model.compile(optimizer=self.optimizer, loss="mse")
        return model

    def build_trainable_graph(self, network: Model) -> Model:
        """build_trainable_graph パラメータ更新のためのネットワークを作成する
        状態と行動を入力とし、Q値を出力するネットワーク

        Args:
            network (Model): 元になるネットワーク(main_network)

        Returns:
            Model: ネットワーク
        """
        action_mask_input = Input(shape=(self.action_len,), name="action_mask_input")
        q_values = network.output
        q_values_taken_action = Dot(axes=-1, name="qs_action")([q_values, action_mask_input])
        trainable_network = Model(inputs=[network.input, action_mask_input], outputs=q_values_taken_action)
        trainable_network.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])
        return trainable_network

    def update_on_batch(self, exps):
        (state, action, reward, next_state, done) = zip(*exps)
        action_index = [self.action_list.index(a) for a in action]
        # 行動をone-hotベクトルに変換したもの
        action_mask = np.array([idx2mask(a, self.action_len) for a in action_index])
        state = np.array(state)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        # max_a Qt(St+1, a)の近似値を計算
        # バッチの状態を入力とし、メインネットワークとターゲットネットワークのQ値のリストを作成する
        next_target_q_values_batch = self.target_network.predict_on_batch(next_state)
        next_q_values_batch = self.main_network.predict_on_batch(next_state)

        # 選択した行動に対応したQ値を近似値として採用する
        if self.double_mode:
            future_return = [
                next_target_q_values[np.argmax(next_q_values)]
                for next_target_q_values, next_q_values in zip(next_target_q_values_batch, next_q_values_batch)
            ]
        else:
            future_return = [np.max(next_q_values) for next_q_values in next_target_q_values_batch]

        # 行動価値関数の目標値を計算
        y = reward + self.gamma * (1 - done) * future_return
        # MSEとlossを使用し、yと出力の差が最小になるようにネットワークを更新
        loss, td_error = self.trainable_network.train_on_batch([state, action_mask], np.expand_dims(y, -1))

        return loss, td_error

    def sync_target_network(self, soft):
        """sync_target_network ターゲットネットワークへの重みの同期

        Args:
            soft ([type]): 重み
        """
        weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= 1 - soft
            target_weights[idx] += soft * w
        self.target_network.set_weights(target_weights)
