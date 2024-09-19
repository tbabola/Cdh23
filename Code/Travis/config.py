import os
from pathlib import Path

save_path = Path("Panels")
data_path = Path(os.getcwd()).parents[0] / 'Data'

color_dict = {'Thy1-GC6s; Cdh23 (Ahl/ahl)':"#DD0000",
                  'Thy1-GC6s; Cdh23 (ahl/ahl)':'#000000',
                  '(F1) Thy1-GC6s; Cdh23 (Ahl/ahl)':'#E69F00'}
order = ['Thy1-GC6s; Cdh23 (ahl/ahl)',
             'Thy1-GC6s; Cdh23 (Ahl/ahl)',
             '(F1) Thy1-GC6s; Cdh23 (Ahl/ahl)']