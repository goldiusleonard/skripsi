from torch.utils.data import IterDataPipe
from sklearn.preprocessing import LabelEncoder


class EncodeLabelIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        lbl_key: str = "type",
    ) -> None:
        self.src_datapipe = src_datapipe
        self.lbl_key = lbl_key
        self.label_encoder = LabelEncoder()
        lbl = []

        for data in src_datapipe:
            lbl.append(data[self.lbl_key])

        self.label_encoder.fit(lbl)

    def __iter__(self):
        for data in self.src_datapipe:
            data[self.lbl_key] = self.label_encoder.transform([data[self.lbl_key]])[0]
            yield data
