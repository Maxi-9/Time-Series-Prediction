import math

import lightning as L
import torch.nn as nn
import torch.utils.data
from overrides import overrides

from TimeSeriesPrediction.model import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LightningBinaryTransformerModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Project input features to model dimension (hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.1)
        # Transformer Encoder: using 4 heads (hidden_size must be divisible by nhead)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dropout=0.1
        )
        encoder_layer.self_attn.batch_first = True
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Final output layer with one output unit
        self.fc = nn.Linear(hidden_size, 1)
        # Sigmoid activation to get output in [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_proj(x)  # -> (batch, seq, hidden_size)
        # Transformer expects input as (seq, batch, hidden_size)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use the output from the last time step
        x = x[-1, :, :]
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        # Use binary cross entropy loss for binary classification
        loss = nn.BCELoss()(outputs, targets.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.BCELoss()(outputs, targets.unsqueeze(1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class BinaryTransformerModel(Commons):
    def __init__(self):
        # Set hyperparameters
        self.hidden_size = 128
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 32
        self.seed = int(time.time() * 1000) % 2**32

        # Define the features used; adjust these as necessary for your dataset
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
        # For binary classification output, we use 1 unit (with sigmoid activation)
        self.model = LightningBinaryTransformerModel(
            input_size,
            self.hidden_size,
            self.num_layers,
            self.learning_rate,
        )

        # Initialize Trainer with checkpointing callback
        self.checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath="checkpoints/",
            filename="binary_transformer_model-{epoch:02d}-{loss:.2f}",
            save_top_k=1,
        )

        self.trainer = L.Trainer(
            max_epochs=self.num_epochs,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=True,
            callbacks=[self.checkpoint_callback],
        )

        super().__init__(self.model, "BinaryTransformer", f_list)

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

        # Reinitialize trainer to reset epoch counter
        self.trainer = L.Trainer(
            max_epochs=self.num_epochs,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=True,
            callbacks=[self.checkpoint_callback],
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
            generator=torch.Generator().manual_seed(self.seed),
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
    def _predict(self, df: pd.DataFrame) -> dict:
        # Return both the binary prediction and the confidence score.
        x_pred = df[self.features.train_cols()].values[-self.lookback :]
        x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_pred)
            confidence = output.cpu().item()
            prediction = 1 if confidence >= 0.5 else 0

        return {"prediction": prediction, "confidence": confidence}


Commons.model_mapping["BinaryTransformer"] = BinaryTransformerModel
