import lightning as L
import torch.nn as nn
import torch.utils.data
from overrides import overrides

from TimeSeriesPrediction.model import *


class LightningSequentialModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x):
        batch_size = x.size(0)
        h0 = self.h0.repeat(1, batch_size, 1)
        c0 = self.c0.repeat(1, batch_size, 1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets.unsqueeze(1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SequentialModel(Commons):
    def __init__(self):
        # Set hyperparameters and initialize model
        self.hidden_size = 128
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 200
        self.batch_size = 32

        feat = [
            Features.Open,
            Features.BB,
            Features.RSI,
            Features.Date,
            Features.MA,
            Features.MACD,
        ]
        f_list = Features(feat, Features.Close)
        input_size = len(list(f_list.train_cols()))
        output_size = 1

        self.model = LightningSequentialModel(
            input_size,
            self.hidden_size,
            self.num_layers,
            output_size,
            self.learning_rate,
        )

        # Initialize Trainer once with checkpointing callback
        self.checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath="checkpoints/",
            filename="model-{epoch:02d}-{loss:.2f}",
            save_top_k=1,
        )

        self.trainer = L.Trainer(
            max_epochs=self.num_epochs,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=True,
            callbacks=[self.checkpoint_callback],
        )

        super().__init__(self.model, "LSTM", f_list)

    @staticmethod
    def worker_init_function(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    @overrides
    def _train(self, df: pd.DataFrame):
        # Prepare data as before
        x, y = Data.train_split(
            df, self.features.train_cols(prev_cols=True), self.features.predict_on
        )
        x_rolled, y_rolled = Data.create_rolling_windows(x, y, self.lookback)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_rolled, dtype=torch.float32),
            torch.tensor(y_rolled, dtype=torch.float32),
        )

        if not self.seed:
            self.seed = int(time.time() * 1000) % 2**32

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
            worker_init_fn=self.worker_init_function,
            generator=torch.Generator().manual_seed(
                self.seed if self.seed is not None else int(time.time() * 1000) % 2**32
            ),
        )

        try:
            self.trainer.fit(self.model, train_loader)
            train_loader.generator = None
        except KeyboardInterrupt:
            print("Stopped training early")

        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, y_test = Data.train_split(
            df, self.features.train_cols(), self.features.predict_on
        )
        x_test_values = x_test.values

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(x_test_values) - self.lookback + 1):
                x_window = x_test_values[i : i + self.lookback]
                x_window = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0)
                output = self.model(x_window)
                predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        return predictions

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        x_pred = df[self.features.train_cols()].values[-self.lookback :]
        x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_pred)
            prediction = output.cpu().item()

        return prediction


Commons.model_mapping["Sequential"] = SequentialModel
