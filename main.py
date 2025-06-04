import scipy.io
import numpy as np
from utils.evaluation import print_results

from algorithms import sso, hoa, ggo, cat_mouse, honey_badger, black_widow, sin_cos_biavoa, icoa, aco_rf, rae, efo, osco, noa, spa, thdoa_gwo_hoa, updated, wroa, hso, gfo, sto, qsso, teso, kmso, dffso, htoa, spo, orpo, tcoa, gdoa, qeoa, thdoa, pis, cro, soso, cdmo, mbo, fno, peso, poa, bfoa, pdoa, bthoa, hsaga, ieho, thdoa_hoa

data = scipy.io.loadmat("data/lungs.mat")
#data = scipy.io.loadmat("data/Lung_Cancer.mat")
X = data["X"]
y = data["Y"].ravel()

algorithms = {
    "SSO": sso,
    "HOA": hoa,
    "GGO": ggo,
    "CatMouse": cat_mouse,
    "HoneyBadger": honey_badger,
    "BlackWidow": black_widow,
    "SinCos-bIAVOA": sin_cos_biavoa,
    "ICOA": icoa,
    "ACO-RF": aco_rf,
    "RAE": rae,
    "EFO": efo,
    "OSCO": osco,
    "NOA": noa,
    "SPA": spa,
    "WROA": wroa,
    "HSO": hso,
    "GFO": gfo,
    "STO": sto,
    "QSSO": qsso,
    "TESO": teso,
    "KMSO": kmso,
    #"DFFSO": dffso,
    #"HTOA": htoa,
    #"SPO": spo,
    "ORPO": orpo,
    "TCOA": tcoa,
    "GDOA": gdoa,
    "QDEA": qeoa,
    "THDOA": thdoa,
    "PIS": pis,
    "CRO": cro,
    #"SoSO": soso,
    "CDMO": cdmo,
    "MBO": mbo,
    "FNO": fno,
    "PESO": peso,
    "POA": poa,
    "BFOA": bfoa,
    "PDOA": pdoa,
    "BTHOA": bthoa,
    #"HSAGA": hsaga,
    "IEHO":ieho,

}
algos = {
   "THDOAHOA":thdoa_hoa,
   "SSO": sso,
   "HOA": hoa
   #"Updated":updated
   #"SSO":sso,
   #"HOA": hoa,
   #"THDOA":thdoa,
   #"THDOAGWOHOA":thdoa_gwo_hoa
   #"IEHO":ieho
}
results = []

for name, module in algos.items():
    try:
        sel, acc, num, t = module.run(X, y)
        print_results(name, sel, acc, num, t)
        results.append((name, acc, num, t))
    except Exception as e:
        print(f"Error in {name}: {e}")
