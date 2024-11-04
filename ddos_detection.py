# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#aici iti alegi calea necesara in functie de ce ai copiat
syn_train_TCP=pd.read_parquet("Syn-training.parquet", engine='pyarrow')
#citesti si fisierul de testare.Iti recomand cel putin RandomForests sa folosesti, are rezultate bune