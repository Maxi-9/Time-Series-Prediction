import time

import lightning as L
import torch.nn as nn
import torch.utils.data
from overrides import overrides

from TimeSeriesPrediction.model import *


class LightningBinarySequentialModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x):
        batch_size = x.size(0)
        # Repeat the initial hidden and cell states for each item in the batch.
        h0 = self.h0.repeat(1, batch_size, 1)
        c0 = self.c0.repeat(1, batch_size, 1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.BCELoss()(outputs, targets.unsqueeze(1).float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.BCELoss()(outputs, targets.unsqueeze(1).float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BinarySequentialModel(Commons):
    def __init__(self):
        # Set hyperparameters.
        self.hidden_size = 128
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 32
        # self.lookback = 300 # Uses default lookback
        self.seed = None

        feat = [
            Features.Open,
            Features.BB,
            Features.RSI,
            Features.Date,
            Features.MA,
            Features.MACD,
        ]
        # Use a target suitable for binary classification.
        f_list = Features(feat, Features.Increased)
        input_features = f_list.train_cols(prev_cols=True)
        input_size = len(list(input_features))
        output_size = 1
        print(
            f"input features, size, out: {input_features}, {input_size}, {output_size}"
        )
        self.model = LightningBinarySequentialModel(
            input_size,
            self.hidden_size,
            self.num_layers,
            output_size,
            self.learning_rate,
        )

        self.checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath="checkpoints/",
            filename="binary_model-{epoch:02d}-{loss:.2f}",
            save_top_k=1,
        )
        self.trainer = L.Trainer(
            max_epochs=self.num_epochs,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=(self.seed is not None),
            callbacks=[self.checkpoint_callback],
        )

        super().__init__(self.model, "BinarySequential", f_list)

    @staticmethod
    def worker_init_function(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np

        np.random.seed(worker_seed)

    @overrides
    def _train(self, df: pd.DataFrame):
        # Reinitialize trainer for a fresh start.
        self.trainer = L.Trainer(
            max_epochs=self.num_epochs,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=(self.seed is not None),
            callbacks=[self.checkpoint_callback],
        )

        x, y = Data.train_split(
            df, self.features.train_cols(prev_cols=True), self.features.predict_on
        )
        x_rolled, y_rolled = Data.create_rolling_windows(x, y, self.lookback)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_rolled, dtype=torch.float32),
            torch.tensor(y_rolled, dtype=torch.float32),
        )
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
        x_test, _ = Data.train_split(
            df, self.features.train_cols(prev_cols=True), self.features.predict_on
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
        x_pred = df[self.features.train_cols(prev_cols=True)].values[-self.lookback :]
        x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_pred)
            prediction = output.cpu().item()
        return prediction


# Register this model.
Commons.model_mapping["BinarySequential"] = BinarySequentialModel
