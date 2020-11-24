import pandas as pd
import numpy as np

import tools

def test_anomalie():
    df = pd.DataFrame(np.random.rand(100))
    media = df.mean()
    dev_std = df.std()
    try:
        res = tools.controlla_anomalie(df.values.reshape(-1), media, dev_std, 2)
    except Exception as e:
        return e, True
    assert res.size == df.size
    try:
        df[res]
    except Exception as e:
        return e, True
    return None, False


def main():
    anomalie, errore = test_anomalie()
    if not errore:
        print("Test anomalie passato.")
    else:
        print("Errore nel test anomalie: {}".format(errore))

if __name__ == "__main__":
    main()