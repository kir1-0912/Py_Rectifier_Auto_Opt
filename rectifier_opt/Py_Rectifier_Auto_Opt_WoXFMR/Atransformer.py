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
        scs_path = f"{os.getcwd()}/spectre_netlist_path1/{short_hash_scs_filename}.scs"
        raw_path = f"{os.getcwd()}/spectre_netlist_path1/{short_hash_scs_filename}.raw/hb.fd.pss_hb"

        # Ð´ netlist
        with open(scs_path, "w") as netlist_file:
            netlist_file.write(spectre_netlist)

        spectre_cmd = f"source /home/imsic/zhangwx/EDA/env_config/env_spectre191_alpsnew_IC618new && spectre {os.getcwd()}/spectre_netlist_path1/{short_hash_scs_filename}.scs -f psfascii"
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
        os.system(f"rm -rf {os.getcwd()}/spectre_netlist_path1/{short_hash_scs_filename}*")


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


def exe_spectre_woxfmr(xnorm: list, RL=3e3, Fre=2.4, Pin=-5):
    C2 = int(xnorm[0])
    N = int(xnorm[1])
    L1 = xnorm[2]
    C1 = int(xnorm[3])
    K = xnorm[4]
    L2 = xnorm[5]
    params_str = f"{C2}_{N}_{L1}_{C1}_{K}_{Fre}_{Pin}_{RL}_{L2}"
    hash_object = hashlib.md5(params_str.encode())
    hash_str = hash_object.hexdigest()
    short_hash_scs_filename = hash_str[:12]
    spectre_netlist = f"""
// Generated for: spectre
// Generated on: Aug 21 10:33:12 2025
// Design library name: rectifier_test
// Design cell name: match_balun_testbench
// Design view name: schematic
simulator lang=spectre
global 0
parameters C1={C1}f C2={C2}f K={K} L1={L1}n L2={L2}n N={N} Pin={Pin} RF={Fre}G RLOAD={RL}
include "/home/share/pdk/tsmc/tsmc65nm/PDK_CRN65LP_v1.7a_Official_IC61_20120914_all/tsmcN65/../models/spectre/toplevel.scs" section=tt_lib

// Library name: rectifier_test
// Cell name: match_balun_testbench
// View name: schematic
PORT1 (VLOAD VSS) port r=RLOAD num=2 type=sine
PORT0 (net4 VSS) port r=50 num=1 type=sine freq=RF dbm=Pin
M34 (net21 net22 net9 VSS) nch_lvt_mac l=60n w=2e-07*N multi=1 nf=N \
        sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*2e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*2e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*2e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*2e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/2e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/2e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
M35 (net22 net21 net9 VSS) nch_lvt_mac l=60n w=2e-07*N multi=1 nf=N \
        sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*2e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*2e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*2e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*2e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/2e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/2e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
M1 (net17 net18 VSS VSS) nch_lvt_mac l=60n w=2e-07*N multi=1 nf=N sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*2e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*2e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*2e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*2e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/2e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/2e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
M0 (net18 net17 VSS VSS) nch_lvt_mac l=60n w=2e-07*N multi=1 nf=N sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*2e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*2e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*2e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*2e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*2e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/2e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/2e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/2e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
C62 (net4 VSS) capacitor c=C1
C53 (net21 RF2) capacitor c=3p
C63 (RF1 RF2) capacitor c=C2
C2 (VLOAD VSS) capacitor c=3p
C52 (RF1 net22) capacitor c=3p
C1 (net18 RF2) capacitor c=3p
C0 (RF1 net17) capacitor c=3p
M36 (net21 net22 VLOAD VLOAD) pch_lvt_mac l=60n w=5e-07*N multi=1 nf=N \
        sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*5e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*5e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*5e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*5e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/5e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/5e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
M37 (net22 net21 VLOAD VLOAD) pch_lvt_mac l=60n w=5e-07*N multi=1 nf=N \
        sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*5e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*5e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*5e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*5e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/5e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/5e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
M3 (net17 net18 net9 net9) pch_lvt_mac l=60n w=5e-07*N multi=1 nf=N \
        sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*5e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*5e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*5e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*5e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/5e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/5e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
M2 (net18 net17 net9 net9) pch_lvt_mac l=60n w=5e-07*N multi=1 nf=N \
        sd=200n \
        ad=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*((N/2)*2e-07))*5e-07 \
        as=((N-int(N/2)*2)*(1.75e-07+((N-1)*2e-07)/2+0)+(N+1-int((N+1)/2)*2)*(1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0))*5e-07 \
        pd=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*(((N/2)*2e-07)*2+N*5e-07) \
        ps=(N-int(N/2)*2)*((1.75e-07+((N-1)*2e-07)/2+0)*2+(N+1)*5e-07)+(N+1-int((N+1)/2)*2)*((1.75e-07+1.75e-07+(N/2-1)*2e-07+0+0)*2+(N+2)*5e-07) \
        nrd=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07/N/5e-07) \
        nrs=(N-int(N/2)*2)*(1e-07*1e-07/(1e-07+1e-07*(N-1))/5e-07)+(N+1-int((N+1)/2)*2)*(1e-07*1e-07*1e-07/(1e-07*1e-07*(N-2)+1e-07*(1e-07+1e-07))/5e-07) \
        sa=175.00n sb=175.00n sca=0 scb=0 scc=0 sigma=1
V0 (VSS 0) vsource dc=0 type=dc
L35 (RF1 RF2) inductor l=L2 q=10 fq=2.4G mode=1
L34 (net4 VSS) inductor l=L1 q=10 fq=2.4G mode=1
K0 mutual_inductor coupling=K ind1=L34 ind2=L35
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
save PORT1:p VLOAD 
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


def find_max_spectre_rl(xnorm, Fre, Pin):
    left = 1  # note 1*500
    right = 200  # note: 200*500
    cache = {}
    max_val = -float('inf')
    best_rl = None

    def evaluate(rl):
        nonlocal max_val, best_rl
        if rl not in cache:
            # print(0)
            val = exe_spectre_woxfmr(xnorm=xnorm, RL=rl * 500, Fre=Fre, Pin=Pin)
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


def round_to_nearest_100(x):
    return round(x / 100) * 100


def Rect_Opt_Flow(xnorm:list):
    try:
        # print(f"Input to Rect_Opt_Flow: {xnorm}")
        Pin_list = [-10, -5, 0, 5]
        Fre_list = [2.4]
        fom_counter = 1
        fom_dict = {}
        rl_dict = {}
        rl_count = 1
        for pin in Pin_list:
            for fre in Fre_list:
                # print(1)
                RL, eta = find_max_spectre_rl(xnorm, Fre=fre, Pin=pin)
                # print(f"f{fre}_P{pin} :{eta}")
                # label_base = f"f{fre}_P{pin}"
                eta_percent = eta * 10
                fom = eta_percent
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
        with open("/home/ieda/wangl/PY/Py_Rectifier_Auto_Opt_WoXFMR/RESULTZY/resulttransformer.csv", mode="a+") as f:
            # f.write(
            #     f"{int(xnorm[0])}, {int(xnorm[1])},{int(xnorm[2])}, {int(xnorm[3])}, "
            #     f"" f"{int(xnorm[4])}, {int(xnorm[5])}, {int(xnorm[6])}, {int(xnorm[7])},"
            #     f" {int(xnorm[8])}, {int(xnorm[9])}, {int(xnorm[10])}, "
            #     f"{xnorm[11]},{xnorm[12]}, {RL_2_4G} , {best_eta_2_4G}, {RL_5G},{best_eta_5G} , {FoM} \n")

            f.write(
                f"{int(xnorm[0])}, {int(xnorm[1])},{xnorm[2]}, {int(xnorm[3])}, "
                f"{xnorm[4]}, {xnorm[5]}, {fom_dict.get('fom1', 1e-12)}, {rl_dict.get('rl1')},"
                f"{fom_dict.get('fom2', 1e-12)}, {rl_dict.get('rl2')},"
                f"{fom_dict.get('fom3', 1e-12)}, {rl_dict.get('rl3')}, {fom_dict.get('fom4', 1e-12)}, {rl_dict.get('rl4')}\n")
    except Exception as e:
        print(f"[simwrong] give{e}")
        return None
    finally:
        print(f"sim finish")

    os.system("rm -rf ./*.log")
    # with open("note.txt",mode="a+") as f2:
    #     f2.write(f"{str(xnorm)} {FoM}")
    # return FoM if not (FoM == None) else 1e20
    fom_list = [fom_dict[f"fom{i}"] for i in range(1, 5)]
    processed_fom_list = [FoM if FoM is not None else 1e-12 for FoM in fom_list]
    return processed_fom_list

# print(Rect_Opt_Flow(
#     [260,9,4.7,2,2,
#      1000,1000,6,30,4,6,30,2,1000,1000]
# ))












