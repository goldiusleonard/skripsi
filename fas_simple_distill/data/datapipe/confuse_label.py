from torch.utils.data import IterDataPipe
from random import randint

class ConfuseLabelIterDataPipe(IterDataPipe):
    def __init__(
        self,
        src_datapipe: IterDataPipe,
        lbl_key: str = "subject",
        num_class: int = 140,
    ) -> None:
        self.src_datapipe = src_datapipe
        self.lbl_key = lbl_key
        self.num_class = num_class
    
    def __iter__(self):
        for data in self.src_datapipe:
            if not f"{self.lbl_key}_confused" in data.keys():
                new_lbl = data[self.lbl_key]
                while new_lbl == data[self.lbl_key]:
                    new_lbl = randint(0, self.num_class-1)
                data[f"{self.lbl_key}_confused"] = new_lbl
            yield data