#!/bin/bash
set -euo pipefail
D="src/walpurgis/datasets"; R="$D/raw_data/METR-LA"; G="$D/sensor_graph"; O="$D/METR-LA"
mkdir -p "$R" "$G" "$O"
echo "[1/3] download..."
if [ ! -f "$R/metr-la.h5" ]||[ "$(stat -c%s $R/metr-la.h5 2>/dev/null||echo 0)" -lt "1000000" ]; then
wget -q "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download" -O /tmp/ml.zip
unzip -o /tmp/ml.zip -d /tmp/; cp /tmp/metr_la.h5 "$R/metr-la.h5"
cp /tmp/distances_la.csv /tmp/sensor_ids_la.txt /tmp/sensor_locations_la.csv "$G/"
rm -f /tmp/ml.zip; else echo "  exists"; fi
echo "[2/3] adj..."
python3 -c "
import pandas as pd,numpy as np,pickle
df=pd.read_hdf('$R/metr-la.h5');ids=[str(c) for c in df.columns];id2i={s:i for i,s in enumerate(ids)}
dd=pd.read_csv('$G/distances_la.csv');n=len(ids);dm=np.zeros((n,n),dtype=np.float64)
for _,r in dd.iterrows():
 fi,ti=str(int(r['from'])),str(int(r['to']))
 if fi in id2i and ti in id2i:dm[id2i[fi],id2i[ti]]=float(r['cost'])
s=dm[dm>0].std();am=np.exp(-dm**2/s**2);am[dm==0]=0;np.fill_diagonal(am,0);am[am<0.1]=0
with open('$G/adj_mx_la.pkl','wb') as f:pickle.dump((ids,id2i,am),f,protocol=2)
print(f'  {am.shape} edges={int((am>0).sum())}')
"
echo "[3/3] npz..."
python3 -c "
import numpy as np,pandas as pd,os
df=pd.read_hdf('$R/metr-la.h5');d=np.expand_dims(df.values,-1)
tod=(df.index.values-df.index.values.astype('datetime64[D]'))/np.timedelta64(1,'D')
tod=np.tile(tod,[1,df.shape[1],1]).transpose((2,1,0))
dow=np.tile(df.index.dayofweek.values.astype(np.float64),[1,df.shape[1],1]).transpose((2,1,0))
d=np.concatenate([d,tod,dow],axis=-1).astype(np.float32)
xo=np.arange(-11,1);yo=np.arange(1,13);idx=np.arange(11,d.shape[0]-12)
x=d[idx[:,None]+xo[None,:]];y=d[idx[:,None]+yo[None,:]]
nt=round(len(x)*0.2);ntr=round(len(x)*0.7);nv=len(x)-nt-ntr
for nm,xd,yd in[('train',x[:ntr],y[:ntr]),('val',x[ntr:ntr+nv],y[ntr:ntr+nv]),('test',x[-nt:],y[-nt:])]:
 np.savez_compressed(f'$O/{nm}.npz',x=xd,y=yd);print(f'  [{nm}] {xd.shape}')
"
echo "done."
