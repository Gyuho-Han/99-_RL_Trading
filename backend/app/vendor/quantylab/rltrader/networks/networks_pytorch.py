import threading
import abc
import numpy as np

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DuelingHead(torch.nn.Module):
    """Dueling DQN 헤드 (오승상 DRL 강의자료 p.82-85).

    공통 특성 벡터를 상태가치 V(s) 스트림과 어드밴티지 A(s,a) 스트림으로 분리한 뒤
    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a) 로 합성한다.
    (mean 차감은 V/A 분해의 식별성(identifiability) 문제 해결용 — 강의자료 p.84-85)
    """

    def __init__(self, in_features, num_actions):
        super().__init__()
        self.value = torch.nn.Linear(in_features, 1)
        self.advantage = torch.nn.Linear(in_features, num_actions)

    def forward(self, x):
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                shared_network=None, activation='sigmoid', loss='mse',
                dueling=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.dueling = dueling

        inp = None
        if hasattr(self, 'num_steps'):
            inp = (self.num_steps, input_dim)
        else:
            inp = (self.input_dim,)

        # 공유 신경망 사용
        self.head = None
        if self.shared_network is None:
            self.head = self.get_network_head(inp, self.output_dim)
            # Dueling 구조: 마지막 Linear 층을 V/A 이중 스트림 헤드로 교체
            if dueling:
                self.head = self._to_dueling(self.head, self.output_dim)
        else:
            self.head = self.shared_network
        
        # 공유 신경망 미사용
        # self.head = self.get_network_head(inp, self.output_dim)

        self.model = torch.nn.Sequential(self.head)
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', torch.nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', torch.nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('activation', torch.nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', torch.nn.Softmax(dim=1))
        self.model.apply(Network.init_weights)
        self.model.to(device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        self.criterion = None
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = torch.nn.BCELoss()

    @staticmethod
    def _to_dueling(head, output_dim):
        """헤드 마지막 Linear 층을 DuelingHead(V+A 스트림)로 교체한다."""
        modules = list(head.children())
        last = modules[-1]
        if not isinstance(last, torch.nn.Linear):
            return head  # 예상 구조가 아니면 그대로 사용
        feature = torch.nn.Sequential(*modules[:-1])
        return torch.nn.Sequential(feature, DuelingHead(last.in_features, output_dim))

    def predict(self, sample):
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(sample).float().to(device)
                pred = self.model(x).detach().cpu().numpy()
                pred = pred.flatten()
            return pred

    def predict_on_batch(self, x):
        """배치 예측 (B, output_dim). DQN 개선 기법의 타깃 계산용."""
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
                return self.model(t).detach().cpu().numpy()

    def train_on_batch(self, x, y, weights=None):
        """배치 학습. weights 가 주어지면 샘플별 가중 MSE (PER 의 IS 보정용)."""
        loss = 0.
        with self.lock:
            self.model.train()
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y_pred = self.model(x)
            if weights is not None:
                w = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(device)
                per_sample = ((y_pred - y) ** 2).mean(dim=1)
                _loss = (w * per_sample).mean()
            else:
                _loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
            loss += _loss.item()
        return loss

    def copy_weights_from(self, other):
        """타깃 네트워크 하드 업데이트: other(행동망)의 가중치를 복사한다."""
        with self.lock:
            self.model.load_state_dict(other.model.state_dict())

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        if net == 'dnn':
            return DNN.get_network_head((input_dim,), output_dim)
        elif net == 'lstm':
            return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'cnn':
            return CNN.get_network_head((num_steps, input_dim), output_dim)

    @abc.abstractmethod
    def get_network_head(inp, output_dim):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init.normal_(weight, std=0.01)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)

    def load_model(self, model_path):
        if model_path is not None:
            # 전체 nn.Module 을 직렬화해 저장하므로 weights_only=False 필요.
            # (자체 생성한 신뢰 가능한 파일) PyTorch 2.6+ 기본값 변경 대응.
            self.model = torch.load(model_path, weights_only=False)
    
class DNN(Network):
    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Linear(inp[0], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y, weights=None):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y, weights=weights)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)

    def predict_on_batch(self, x):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().predict_on_batch(x)


class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y, weights=None):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y, weights=weights)

    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
        return super().predict(sample)

    def predict_on_batch(self, x):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().predict_on_batch(x)


class LSTMModule(torch.nn.LSTM):
    def __init__(self, *args, use_last_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_last_only = use_last_only

    def forward(self, x):
        output, (h_n, _) = super().forward(x)
        if self.use_last_only:
            return h_n[-1]
        return output


class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Conv1d(inp[0], 1, kernel_size),
            torch.nn.BatchNorm1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(inp[1] - (kernel_size - 1), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y, weights=None):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y, weights=weights)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)

    def predict_on_batch(self, x):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().predict_on_batch(x)
