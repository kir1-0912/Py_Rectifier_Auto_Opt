import os
import re
import hashlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from Balun_X_Example_Single_Gen import Balun_X_Gen
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import threading
import math
import os, subprocess


def run_spectre_safe(short_hash_scs_filename, spectre_netlist, Pin):
    try:
        cwd = os.getcwd()
        scs_path = f"{os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.scs"
        raw_path = f"{os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.raw/hb.fd.pss_hb"

        # Ð´ netlist
        with open(scs_path, "w") as netlist_file:
            netlist_file.write(spectre_netlist)

        spectre_cmd = f"source /home/imsic/zhangwx/EDA/env_config/env_spectre191_alpsnew_IC618new && spectre {os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.scs -f psfascii"
        result = subprocess.run(spectre_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode != 0:
            raise RuntimeError(f"returncode: {result.returncode} file: {scs_path}")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"no file: {raw_path}")
        with open(raw_path, "r") as psf:
            psf_content = psf.read()
        parsed_data = parse_freq_data(psf_content)
        eta_total = (parsed_data[0]["VLOAD"] * parsed_data[0]["PORT1:p"]).real / dBm2w(Pin)
        # print(eta_total)
        return eta_total
    except Exception as e:
        # print(f"[simwrong] give{e}")
        return 1e-12
    finally:
        os.system(f"rm -rf {os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}*")





def parse_freq_data(text):
    block_pattern = re.compile(
        r'(?i)"freq"\s+([\d.e+-]+)\s*((?:.*?(?=\s*"freq"|\s*END|\s*end|\Z))+)',
        re.DOTALL
    )
    blocks = []
    for match in block_pattern.finditer(text):
        freq = float(match.group(1))
        content = match.group(2).strip()
        blocks.append((freq, content))
    result = []
    for freq, content in blocks:
        current = {"freq": freq}
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # print(line)
            key = line.strip("\"").split("\"")[0].strip()
            valuestr = line.strip("\"").split("\"")[1].strip().strip("(").strip(")")
            value = complex(float(valuestr.split()[0]), float(valuestr.split()[1]))
            current[key] = value
        result.append(current)
    return result


def dBm2w(PindBm):
    return 10 ** (float(PindBm) / 10) / 1000


def exe_spectre_woxfmr(xnorm: list, RLOAD=3e3, RF=2.4, Pin=-5):
    C1 = xnorm[0]
    C2 = xnorm[1]
    L1 = xnorm[2]
    N = int(xnorm[3])
    L2 = xnorm[4]
    params_str = f"{C1}_{C2}_{L1}_{N}_{RF}_{Pin}_{RLOAD}"
    hash_object = hashlib.md5(params_str.encode())
    hash_str = hash_object.hexdigest()
    short_hash_scs_filename = hash_str[:12]
    spectre_netlist = f"""
// Generated for: spectre
// Generated on: Sep 23 10:10:54 2025
// Design library name: rect
// Design cell name: wideband_test1
// Design view name: schematic
simulator lang=spectre
global 0
parameters C1={C1}p C2={C2}p L1={L1}n L2={L2}n RF={RF}G N={N} RLOAD={RLOAD} Pin={Pin}
include "/home/share/pdk/tsmc/tsmc28nm/T28HPC_20250311/PDK/iPDK_CLN28HPC+_v1.0_2p2a_20150612_all/iPDK_CLN28HPC+_v1.0_2p2a_20150612/tsmcN28/../models/spectre/toplevel.scs" section=top_tt
include "/home/imsic/zhangwx/pdk/tsmc/tsmcN28/iPDK_CRN28HPC+_v1.0_2p2a_20170531_all/models/spectre/crn28hpcp_1d8_elk_v1d0_2p2.scs" section=tt_rfind

// Library name: analogLib
// Cell name: ideal_balun
// View name: schematic
subckt ideal_balun d c p n
    K0 (d 0 p c) transformer n1=2
    K1 (d 0 c n) transformer n1=2
ends ideal_balun
// End of subcircuit definition.

// Library name: rect
// Cell name: wideband_test1
// View name: schematic
M1 (net7 net8 VSS VSS) nch_lvt_mac l=30n w=1e-06*N multi=1 nf=N sd=100n \
        ad=((N-int(N/2)*2)*(7.5e-08+((N-1)*1e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*1e-07))*1e-06 \
        as=((N-int(N/2)*2)*(7.5e-08+((N-1)*1e-07)/2+0)+(N+1-int((N+1)/2)*2)*(7.5e-08+7.5e-08+(N/2-1)*1e-07+0+0))*1e-06 \
        pd=(N-int(N/2)*2)*((7.5e-08+((N-1)*1e-07)/2+0)*2+(N+1)*1e-06)+(N+1-int((N+1)/2)*2)*(((N/2)*1e-07)*2+N*1e-06) \
        ps=(N-int(N/2)*2)*((7.5e-08+((N-1)*1e-07)/2+0)*2+(N+1)*1e-06)+(N+1-int((N+1)/2)*2)*((7.5e-08+7.5e-08+(N/2-1)*1e-07+0+0)*2+(N+2)*1e-06) \
        sa=75.0n sb=75.0n sa1=75.0n sa2=75.0n sa3=75.0n sa4=75.0n \
        sb1=75.0n sb2=75.0n sb3=75.0n spa=100n spa1=100n spa2=100n \
        spa3=100n sap=91.9776n sapb=114.444n spba=115.715n spba1=117.043n \
        dfm_flag=0 spmt=1.11111e+15 spomt=0 spomt1=1.11111e+60 \
        spmb=1.11111e+15 spomb=0 spomb1=1.11111e+60
M0 (net8 net7 VSS VSS) nch_lvt_mac l=30n w=1e-06*N multi=1 nf=N sd=100n \
        ad=((N-int(N/2)*2)*(7.5e-08+((N-1)*1e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*1e-07))*1e-06 \
        as=((N-int(N/2)*2)*(7.5e-08+((N-1)*1e-07)/2+0)+(N+1-int((N+1)/2)*2)*(7.5e-08+7.5e-08+(N/2-1)*1e-07+0+0))*1e-06 \
        pd=(N-int(N/2)*2)*((7.5e-08+((N-1)*1e-07)/2+0)*2+(N+1)*1e-06)+(N+1-int((N+1)/2)*2)*(((N/2)*1e-07)*2+N*1e-06) \
        ps=(N-int(N/2)*2)*((7.5e-08+((N-1)*1e-07)/2+0)*2+(N+1)*1e-06)+(N+1-int((N+1)/2)*2)*((7.5e-08+7.5e-08+(N/2-1)*1e-07+0+0)*2+(N+2)*1e-06) \
        sa=75.0n sb=75.0n sa1=75.0n sa2=75.0n sa3=75.0n sa4=75.0n \
        sb1=75.0n sb2=75.0n sb3=75.0n spa=100n spa1=100n spa2=100n \
        spa3=100n sap=91.9776n sapb=114.444n spba=115.715n spba1=117.043n \
        dfm_flag=0 spmt=1.11111e+15 spomt=0 spomt1=1.11111e+60 \
        spmb=1.11111e+15 spomb=0 spomb1=1.11111e+60
M3 (net8 net7 VLOAD VLOAD) pch_lvt_mac l=30n w=1e-06*1.2*N multi=1 \
        nf=1.2*N sd=100n \
        ad=((1.2*N-int(1.2*N/2)*2)*(7.5e-08+((1.2*N-1)*1e-07)/2+0)+(1.2*N+1-int((1.2*N+1)/2)*2)*((1.2*N/2)*1e-07))*1e-06 \
        as=((1.2*N-int(1.2*N/2)*2)*(7.5e-08+((1.2*N-1)*1e-07)/2+0)+(1.2*N+1-int((1.2*N+1)/2)*2)*(7.5e-08+7.5e-08+(1.2*N/2-1)*1e-07+0+0))*1e-06 \
        pd=(1.2*N-int(1.2*N/2)*2)*((7.5e-08+((1.2*N-1)*1e-07)/2+0)*2+(1.2*N+1)*1e-06)+(1.2*N+1-int((1.2*N+1)/2)*2)*(((1.2*N/2)*1e-07)*2+1.2*N*1e-06) \
        ps=(1.2*N-int(1.2*N/2)*2)*((7.5e-08+((1.2*N-1)*1e-07)/2+0)*2+(1.2*N+1)*1e-06)+(1.2*N+1-int((1.2*N+1)/2)*2)*((7.5e-08+7.5e-08+(1.2*N/2-1)*1e-07+0+0)*2+(1.2*N+2)*1e-06) \
        nrd=0.001285 nrs=0.001285 sa=75.0n sb=75.0n sa1=75.0n sa2=75.0n \
        sa3=75.0n sa4=75.0n sb1=75.0n sb2=75.0n sb3=75.0n spa=100n \
        spa1=100n spa2=100n spa3=100n sap=91.9776n sapb=114.444n \
        spba=115.715n spba1=117.043n dfm_flag=0 spmt=1.11111e+15 spomt=0 \
        spomt1=1.11111e+60 spmb=1.11111e+15 spomb=0 spomb1=1.11111e+60
M2 (net7 net8 VLOAD VLOAD) pch_lvt_mac l=30n w=1e-06*1.2*N multi=1 \
        nf=1.2*N sd=100n \
        ad=((1.2*N-int(1.2*N/2)*2)*(7.5e-08+((1.2*N-1)*1e-07)/2+0)+(1.2*N+1-int((1.2*N+1)/2)*2)*((1.2*N/2)*1e-07))*1e-06 \
        as=((1.2*N-int(1.2*N/2)*2)*(7.5e-08+((1.2*N-1)*1e-07)/2+0)+(1.2*N+1-int((1.2*N+1)/2)*2)*(7.5e-08+7.5e-08+(1.2*N/2-1)*1e-07+0+0))*1e-06 \
        pd=(1.2*N-int(1.2*N/2)*2)*((7.5e-08+((1.2*N-1)*1e-07)/2+0)*2+(1.2*N+1)*1e-06)+(1.2*N+1-int((1.2*N+1)/2)*2)*(((1.2*N/2)*1e-07)*2+1.2*N*1e-06) \
        ps=(1.2*N-int(1.2*N/2)*2)*((7.5e-08+((1.2*N-1)*1e-07)/2+0)*2+(1.2*N+1)*1e-06)+(1.2*N+1-int((1.2*N+1)/2)*2)*((7.5e-08+7.5e-08+(1.2*N/2-1)*1e-07+0+0)*2+(1.2*N+2)*1e-06) \
        nrd=0.001285 nrs=0.001285 sa=75.0n sb=75.0n sa1=75.0n sa2=75.0n \
        sa3=75.0n sa4=75.0n sb1=75.0n sb2=75.0n sb3=75.0n spa=100n \
        spa1=100n spa2=100n spa3=100n sap=91.9776n sapb=114.444n \
        spba=115.715n spba1=117.043n dfm_flag=0 spmt=1.11111e+15 spomt=0 \
        spomt1=1.11111e+60 spmb=1.11111e+15 spomb=0 spomb1=1.11111e+60
V0 (VSS 0) vsource dc=0 type=dc
C29 (net1 net3) capacitor c=C2
C28 (net4 net8) capacitor c=3p
C27 (net9 net7) capacitor c=3p
C26 (RF2 net3) capacitor c=C1
C25 (net2 net1) capacitor c=C1
C2 (VLOAD VSS) capacitor c=50p
PORT1 (VLOAD VSS) port r=RLOAD type=sine
PORT0 (net6 VSS) port r=50 num=1 type=sine freq=RF dbm=Pin
I7 (net6 VSS RF1 RF2) ideal_balun
R0 (RF1 net2) resistor r=0
L2 (net3 net4) inductor l=L2 q=10 fq=2.4G mode=1
L1 (net1 net9) inductor l=L2 q=10 fq=2.4G mode=1
L0 (net1 net3) inductor l=L1 q=10 fq=2.4G mode=1
simulatorOptions options psfversion="1.1.0" reltol=1e-3 vabstol=1e-6 \
    iabstol=1e-12 temp=27 tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 \
    maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 \
    sensfile="../psf/sens.output" checklimitdest=psf 
hb  hb  autoharms=yes  autotstab=yes  oversample=[10]
+   fundfreqs=[(RF)]  maxharms=[5]  errpreset=conservative  annotate=status
modelParameter info what=models where=rawfile
element info what=inst where=rawfile
outputParameter info what=output where=rawfile
designParamVals info what=parameters where=rawfile
primitives info what=primitives where=rawfile
subckts info what=subckts where=rawfile
save VLOAD PORT1:p M2:1 M0:3 PORT1:p R0:1 
saveOptions options save=allpub


"""
    eta_total = run_spectre_safe(short_hash_scs_filename, spectre_netlist, Pin)
    # with open(f"{os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.scs",mode="w") as netlist_file:
    #     netlist_file.write(spectre_netlist)
    # spectre_cmd=f"source /home/imsic/zhangwx/EDA/env_config/env_spectre191_alpsnew_IC618new && spectre {os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.scs -f psfascii"
    # # os.system(spectre_cmd)
    # subprocess.run(spectre_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # with open( f"{os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.raw/hb.fd.pss_hb",mode="r") as psf:
    #     psf_content=psf.read()
    # parsed_data = parse_freq_data(psf_content)
    # # print(parsed_data[0]["Vout"])
    # eta_total = (parsed_data[0]["Vout"] * parsed_data[0]["RLOAD:1"]).real / dBm2w(Pin)
    # os.system(f"rm -rf {os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}*")
    return eta_total


def find_max_spectre_rl(xnorm, RF, Pin):
    left = 1  # note 1*500
    right = 20  # note: 200*500
    cache = {}
    max_val = -float('inf')
    best_rl = None

    def evaluate(rl):
        nonlocal max_val, best_rl
        if rl not in cache:
            # print(0)
            val = exe_spectre_woxfmr(xnorm=xnorm, RLOAD=rl * 500, RF=RF, Pin=Pin)
            # print(f"{rl}:{val}")
            cache[rl] = val
            if val > max_val:
                max_val = val
                best_rl = rl
        return cache[rl]

    evaluate(left)
    evaluate(right)
    while right - left > 3:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        if m1 == left: m1 += 1
        if m2 == right: m2 -= 1
        if m1 >= m2: break
        v1 = evaluate(m1)
        v2 = evaluate(m2)
        if v1 < v2:
            left = m1
        else:
            right = m2
    for rl in range(left, right + 1):
        evaluate(rl)
    return best_rl, max_val




def Rect_Opt_Flow(xnorm:list):
    try:
        # print(f"Input to Rect_Opt_Flow: {xnorm}")
        Pin_list = [0]
        Fre_list = [2.4, 5.8]
        fom_counter = 1
        fom_dict = {}
        rl_dict = {}
        rl_count = 1
        for pin in Pin_list:
            for fre in Fre_list:
                # print(1)
                RL, eta = find_max_spectre_rl(xnorm, RF=fre, Pin=pin)
                # print(f"f{fre}_P{pin} :{eta}")
                # label_base = f"f{fre}_P{pin}"
                eta_percent = eta * 10
                fom = eta_percent
                # fom = math.log10(eta_percent)
                fom_name = f"fom{fom_counter}"
                fom_dict[fom_name] = fom
                fom_counter += 1
                rl_name = f"rl{rl_count}"
                rl_dict[rl_name] = RL
                rl_count += 1
        # RL_2_3G, best_eta_2_8G = find_max_spectre_rl(xnorm, Fre=2.8)
        # RL_2_4G, best_eta_3G = find_max_spectre_rl(xnorm,Fre=3)
        # RL_2_3G, best_eta_3_2G = find_max_spectre_rl(xnorm, Fre=3.2)
        # RL_2_2G, best_eta_1_8G = find_max_spectre_rl(xnorm, Fre=1.8)
        # RL_2_2G, best_eta_2G = find_max_spectre_rl(xnorm, Fre=2)
        # RL_2_2G, best_eta_2_2G = find_max_spectre_rl(xnorm, Fre=2.2)
        # RL_5G, best_eta_5G = find_max_spectre_rl(xnorm, Fre=5)
        # FoM= 1/(best_eta_2_4G*100) + 1/(best_eta_5G*100)
        # FoM18 = np.log10(best_eta_1_8G * 100)
        # FoM2 = np.log10(best_eta_2G * 100)
        # FoM22 = np.log10(best_eta_2_2G * 100)
        # FoM28 = np.log10(best_eta_2_8G * 100)
        # FoM3 = np.log10(best_eta_3G * 100)
        # FoM32 = np.log10(best_eta_3_2G * 100)
        with open("/home/ieda/wangl/PY/Py_Rectifier_Auto_Opt_WoXFMR/RESULTZY/resultWB1.csv", mode="a+") as f:
            # f.write(
            #     f"{int(xnorm[0])}, {int(xnorm[1])},{int(xnorm[2])}, {int(xnorm[3])}, "
            #     f"" f"{int(xnorm[4])}, {int(xnorm[5])}, {int(xnorm[6])}, {int(xnorm[7])},"
            #     f" {int(xnorm[8])}, {int(xnorm[9])}, {int(xnorm[10])}, "
            #     f"{xnorm[11]},{xnorm[12]}, {RL_2_4G} , {best_eta_2_4G}, {RL_5G},{best_eta_5G} , {FoM} \n")

            f.write(
                f"{xnorm[0]}, {xnorm[1]},{xnorm[2]}, {int(xnorm[3])}, "
                f"{xnorm[4]},{fom_dict.get('fom1', 0)}, {rl_dict.get('rl1')},"
                f"{fom_dict.get('fom2', 0)}, {rl_dict.get('rl2')}\n")
    except Exception as e:
        print(f"[simwrong] give{e}")
        return None
    finally:
        print(f"sim finish")

    os.system("rm -rf ./*.log")
    # with open("note.txt",mode="a+") as f2:
    #     f2.write(f"{str(xnorm)} {FoM}")
    # return FoM if not (FoM == None) else 1e20
    fom_list = [fom_dict[f"fom{i}"] for i in range(1, 3)]
    processed_fom_list = [FoM if FoM is not None else 0 for FoM in fom_list]
    return processed_fom_list

# print(Rect_Opt_Flow(
#     [260,9,4.7,2,2,
#      1000,1000,6,30,4,6,30,2,1000,1000]
# ))


# xnorm = [1844, 397, 0.593810, 1334, 4870, 204]
# RF = 2.4
# Pin = 0
# a,b=find_max_spectre_rl(xnorm, RF, Pin)
# print(a)








