"""Verify both saved VAE state dicts on the fixed held-out split."""
import ast,argparse,hashlib,json
from pathlib import Path
import numpy as np,torch,joblib
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent;nb=json.loads((root/'analysis.ipynb').read_text());scope={'torch':torch,'nn':nn}
for cell in nb['cells']:
 if cell['cell_type']=='code':
  tree=ast.parse(''.join(cell['source']));nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VAE']
  if nodes:exec(compile(ast.Module(nodes,type_ignores=[]),str(root/'analysis.ipynb'),'exec'),scope)
d=load_breast_cancer();train,test=train_test_split(np.arange(len(d.target)),test_size=.2,random_state=42,stratify=d.target)
x=d.data.astype(np.float32);scaler=joblib.load(a.run/'scaler.joblib');xt=torch.tensor(scaler.transform(x[test]).astype(np.float32));metrics=json.loads((a.run/'metrics.json').read_text());out={}
for name,key in [('vae_a','mse_a'),('vae_b','mse_b')]:
 model=scope['VAE']();model.load_state_dict(torch.load(a.run/(name+'.pt'),map_location='cpu',weights_only=True),strict=True);model.eval()
 with torch.no_grad():value=float(nn.functional.mse_loss(model.decode(model.encode(xt)[0]),xt))
 if not np.isclose(value,metrics[key],atol=1e-7,rtol=1e-6):raise AssertionError('Reloaded test MSE differs')
 out[name]={'mse':value,'checkpoint_sha256':hashlib.sha256((a.run/(name+'.pt')).read_bytes()).hexdigest(),'strict_load':True}
(a.run/'checkpoint-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
