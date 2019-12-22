import numpy as np
import pandas as pd

import settings
from src.plsa import PLSA

if __name__=="__main__":
    #多項分布データセットの読み込み
    multi = pd.read_csv(
        settings.dataset,
        header=settings.dataset_header,
        )


    #別ファイルのplsa.pyを用いてPLSAを実行
    plsa = PLSA(multi, settings.Z)
    plsa.train()