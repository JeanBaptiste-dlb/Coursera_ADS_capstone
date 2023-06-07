from typing import List

import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from app.src.models.architecture import Net

input_columns = [
    "Page total likes",
    "Type",
    "Category",
    "Paid",
    "Post Month",
    "Post Weekday",
    "Post Hour",
    "Lifetime Post Total Reach",
    "Lifetime Post Total Impressions",
    "Lifetime Engaged Users",
    "Lifetime Post Consumers",
    "Lifetime Post Consumptions",
    "Lifetime Post Impressions by people who have liked your Page",
    "Lifetime Post reach by people who like your Page",
    "Lifetime People who have liked your Page and engaged with your post",
]
columns_to_threshold = [
    "Page total likes",
    "Lifetime Post Total Reach",
    "Lifetime Post Total Impressions",
    "Lifetime Engaged Users",
    "Lifetime Post Consumers",
    "Lifetime Post Consumptions",
    "Lifetime Post Impressions by people who have liked your Page",
    "Lifetime Post reach by people who like your Page",
    "Lifetime People who have liked your Page and engaged with your post",
]
columns_to_scale = [
    "Page total likes",
    "Post Month",
    "Post Weekday",
    "Post Hour",
    "Lifetime Post Total Reach",
    "Lifetime Post Total Impressions",
    "Lifetime Engaged Users",
    "Lifetime Post Consumers",
    "Lifetime Post Consumptions",
    "Lifetime Post Impressions by people who have liked your Page",
    "Lifetime Post reach by people who like your Page",
    "Lifetime People who have liked your Page and engaged with your post",
]
output_col_order = [
    "Link",
    "Photo",
    "Status",
    "Video",
    "category_1",
    "category_2",
    "category_3",
    "Page total likes",
    "Post Month",
    "day_1",
    "day_2",
    "day_3",
    "day_4",
    "day_5",
    "day_6",
    "day_7",
    "Post Hour",
    "Paid",
    "Lifetime Post Total Reach",
    "Lifetime Post Total Impressions",
    "Lifetime Engaged Users",
    "Lifetime Post Consumers",
    "Lifetime Post Consumptions",
    "Lifetime Post Impressions by people who have liked your Page",
    "Lifetime Post reach by people who like your Page",
    "Lifetime People who have liked your Page and engaged with your post",
]
n_features = torch.load("./models/best_net")["model.0.weight"].shape[1]
path_to_scaler_ds = "./data/raw/dataset_Facebook.csv"
model = Net(n_features)
model.load_state_dict(torch.load("./models/best_net"))


class AppPipeline:
    def __init__(
        self,
        input_columns: List = input_columns,
        output_col_order: List = output_col_order,
        columns_to_threshold: List = columns_to_threshold,
        path_to_scaler_ds: str = path_to_scaler_ds,
        predictor=model,
        batch_size=20,
    ) -> None:
        self.input_columns = input_columns
        self.columns_to_threshold = columns_to_threshold
        self.columns_to_scale = columns_to_scale
        self.output_col_order = output_col_order
        self.path_to_scaler_ds = path_to_scaler_ds
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = predictor
        self.model.to(self.device)

    def __call__(self, inputs_: List[dict]) -> List:
        processed_input = self.scale_incoming_sample(inputs_)
        processed_input = torch.Tensor(np.array(processed_input))
        return self.predict(processed_input)

    def predict(self, scaled_inputs):
        dataset_train = data.TensorDataset(scaled_inputs)
        dl = data.DataLoader(dataset_train, batch_size=self.batch_size)
        self.model.eval()
        ress = []
        for i, batch_x in tqdm(enumerate(dl)):
            with torch.no_grad():
                y = self.model(batch_x[0].to(self.device))
            ress.append(y.detach().cpu().numpy())
        return np.vstack(ress)[:, 0]

    def scale_incoming_sample(
        self,
        inputs: List[dict],
        columns_to_threshold=columns_to_threshold,
        columns_to_scale=columns_to_scale,
        output_col_order=output_col_order,
    ):
        inputs = pd.DataFrame(inputs)
        n_samples = len(inputs)
        data_for_scale = pd.read_csv(self.path_to_scaler_ds, sep=";")
        base_for_scale = data_for_scale[input_columns]
        to_scale = pd.concat([inputs, base_for_scale])
        to_scale = pd.concat([pd.get_dummies(to_scale.Category), to_scale], axis=1)
        to_scale = pd.concat([pd.get_dummies(to_scale.Type), to_scale], axis=1)
        to_scale = to_scale.rename(
            columns={1: "category_1", 2: "category_2", 3: "category_3"}
        )
        to_scale = pd.concat(
            [pd.get_dummies(to_scale["Post Weekday"]), to_scale], axis=1
        )
        to_scale = to_scale.rename(
            columns={
                1: "day_1",
                2: "day_2",
                3: "day_3",
                4: "day_4",
                5: "day_5",
                6: "day_6",
                7: "day_7",
            }
        )
        to_scale = to_scale.drop(columns=["Type", "Category"])
        for col in columns_to_threshold:
            threshold = np.percentile(to_scale[col], 95)
            to_scale[col] = to_scale.apply(lambda x: min(x[col], threshold), axis=1)
        scaler = preprocessing.MinMaxScaler()
        to_scale[columns_to_scale] = scaler.fit_transform(to_scale[columns_to_scale])
        scaled_inputs = to_scale[:n_samples]
        return scaled_inputs[output_col_order]


app = AppPipeline(predictor=model)
