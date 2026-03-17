import scipy.io
import numpy as np
from utils.evaluation import print_results

from algorithms import dmfs, gcfs, gwo, gwo_hoa, gwo_new, ieho_new, npo, psocsm, sso, hoa, ggo, cat_mouse, honey_badger, black_widow, sin_cos_biavoa, icoa, aco_rf, rae, efo, osco, noa, spa, thdoa_amgwo, thdoa_cgwo, thdoa_cmwgwo, thdoa_ebgwo, thdoa_gwo, thdoa_gwo_hoa, thdoa_gwo_new, thdoa_gwo_new2, thdoa_gwo_new3, thdoa_gwo_new4, thdoa_gwo_stats, thdoa_hoa_new, thdoa_iagwo, thdoa_ieho, thdoa_igwo, thdoa_mgwo, thdoa_pgwocsa, thdoa_sca, thdoasso, updated, wroa, hso, gfo, sto, qsso, teso, kmso, dffso, htoa, spo, orpo, tcoa, gdoa, qeoa, thdoa, pis, cro, soso, cdmo, mbo, fno, peso, poa, bfoa, pdoa, bthoa, hsaga, ieho, thdoa_hoa

# data = scipy.io.loadmat("data/Brain_Tumor_1.mat")
# data = scipy.io.loadmat("data/CLL_SUB_111.mat")
# data = scipy.io.loadmat("data/DLBCL.mat")
# data = scipy.io.loadmat("data/GLI_85.mat")
# data = scipy.io.loadmat("data/Leukemia_1.mat")
# data = scipy.io.loadmat("data/Leukemia_3.mat")
# data = scipy.io.loadmat("data/Lung_Cancer.mat")
# data = scipy.io.loadmat("data/lungs.mat")
# data = scipy.io.loadmat("data/nci9.mat")
# data = scipy.io.loadmat("data/Prostate_Tumor_1.mat")
data = scipy.io.loadmat("data/SMK_CAN_187.mat")
# data = scipy.io.loadmat("data/TOX_171.mat")

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
    "Updated":updated,
    "THDOASSO": thdoasso,
    "THDOAGWOHOA":thdoa_gwo_hoa,
    "THDOAIEHO":thdoa_ieho,
    "THDOAHOA": thdoa_hoa,
    "THDOAHOANEW": thdoa_hoa_new,
    "PSOCSM":psocsm,
    "NPO":npo,
    #"GCFS": gcfs,
    "DMFS":dmfs,
    "GWO": gwo,
    "GWONEW": gwo_new,
    "THDOAGWO":thdoa_gwo,
    "GWOHOA": gwo_hoa,
    "THDOASCA": thdoa_sca,
    "THDOAGWONEW":thdoa_gwo_new,
    "THDOAGWONEW2":thdoa_gwo_new2,
    "THDOAGWONEW3":thdoa_gwo_new3,
    "THDOAGWO":thdoa_gwo_stats,
    "THDOAiGWO":thdoa_igwo,
    "THDOAmGWO":thdoa_mgwo,
    "THDOAcGWO":thdoa_cgwo,
    "THDOAEBGWO":thdoa_ebgwo,
    "THDOAPGWOCSA":thdoa_pgwocsa,
    "THDOAIAGWO":thdoa_iagwo,
    "THDOACMWGWO":thdoa_cmwgwo
}
algos = {
    #"THDOAGWO":thdoa_gwo_stats
   #"THDOAGWO":thdoa_gwo_new3
   "THDOAAMGWO":thdoa_amgwo
}
results = []

for name, module in algos.items():
    try:
        sel, acc, num, t = module.run(X, y)
        print_results(name, sel, acc, num, t)
        results.append((name, acc, num, t))
    except Exception as e:
        print(f"Error in {name}: {e}")
