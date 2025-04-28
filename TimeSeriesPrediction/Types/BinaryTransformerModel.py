import time

import lightning as L
import torch.nn as nn
import torch.utils.data
from overrides import overrides

from TimeSeriesPrediction.model import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return x


class LightningBinaryTransformerModel(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        output_size: int,
        learning_rate: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        # project your inputs into the transformer dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # add positional information
        self.pos_encoder = PositionalEncoding(d_model)

        # the actual transformer stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # using seq_len, batch order
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # **this** must live inside __init__ at the same indent!
        self.fc_out = nn.Linear(d_model, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # → (seq_len, batch, d_model)
        x = self.pos_encoder(x)  # add sin/cos
        enc_out = self.transformer_encoder(x)
        last = enc_out[-1]  # grab final timestep → (batch, d_model)
        logits = self.fc_out(last)  # now works!
        return self.sigmoid(logits)

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
        return torch.optim.Adam(self.parameters(), lr=0.001)


class BinaryTransformerModel(Commons):
    def __init__(self):
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 2
        self.dim_feedforward = 256
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 32
        self.seed = None

        feat = [
            Features.Open,
            Features.BB,
            Features.RSI,
            Features.Date,
            Features.MA,
            Features.MACD,
        ]
        f_list = Features(feat, Features.Increased)
        input_features = f_list.train_cols(prev_cols=True)
        input_size = len(list(input_features))
        output_size = 1

        self.model = LightningBinaryTransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            output_size=output_size,
            learning_rate=self.learning_rate,
        )

        self.checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath="checkpoints/",
            filename="binary_transformer-{epoch:02d}-{loss:.2f}",
            save_top_k=1,
        )
        self.trainer = L.Trainer(
            max_epochs=self.num_epochs,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=(self.seed is not None),
            callbacks=[self.checkpoint_callback],
        )

        super().__init__(self.model, "BinaryTransformer", f_list)

    @staticmethod
    def worker_init_function(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np

        np.random.seed(worker_seed)

    @overrides
    def _train(self, df: pd.DataFrame):
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

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_rolled, dtype=torch.float32),
            torch.tensor(y_rolled, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
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
            self.trainer.fit(self.model, loader)
        except KeyboardInterrupt:
            print("Stopped training early")
        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, _ = Data.train_split(
            df, self.features.train_cols(prev_cols=True), self.features.predict_on
        )
        x_vals = x_test.values

        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(len(x_vals) - self.lookback + 1):
                window = x_vals[i : i + self.lookback]
                tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                preds.append(self.model(tensor).cpu().numpy())
        return np.concatenate(preds, axis=0)

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        arr = df[self.features.train_cols(prev_cols=True)].values[-self.lookback :]
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor).cpu().item()


# Register the transformer variant
Commons.model_mapping["BinaryTransformer"] = BinaryTransformerModel
