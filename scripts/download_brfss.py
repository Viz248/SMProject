"""Download the exact public CDC releases used by the availability study."""
import argparse,hashlib,shutil,urllib.request,zipfile
from pathlib import Path
RELEASES={
 '2024':('https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip','d33de91a584e866f657d29b1d5b51b5672467286ed22d021e36ee3c7b4d712a4','7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c'),
 '2025':('https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip','d26a8b391aa2966b58dc2a3ffddf8da050e9f38eb143d0f5ca7b3d21ece08f98','d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2')}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--years',nargs='+',choices=RELEASES,default=['2024','2025']);a=p.parse_args()
 root=Path(a.directory);root.mkdir(parents=True,exist_ok=True)
 for year in a.years:
  url,ziphash,xpthash=RELEASES[year];archive=root/f'LLCP{year}XPT.zip';target=root/f'LLCP{year}.XPT'
  if target.exists():
   if digest(target)!=xpthash:raise ValueError(f'Existing XPT does not match frozen release: {target}')
   print('Verified',target);continue
  if not archive.exists():
   temporary=archive.with_suffix('.download')
   with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as f:shutil.copyfileobj(response,f)
   if digest(temporary)!=ziphash:raise ValueError('CDC archive hash differs from frozen release; review version rather than silently updating')
   temporary.replace(archive)
  if digest(archive)!=ziphash:raise ValueError('Archive hash mismatch')
  with zipfile.ZipFile(archive) as z:
   members=[n for n in z.namelist() if Path(n).name.strip().upper()==f'LLCP{year}.XPT']
   if len(members)!=1:raise ValueError('Unexpected archive layout')
   temporary=target.with_suffix('.extracting')
   with z.open(members[0]) as source,temporary.open('wb') as f:shutil.copyfileobj(source,f)
   if digest(temporary)!=xpthash:raise ValueError('Extracted XPT hash mismatch')
   temporary.replace(target)
  print('Downloaded and verified',target)
if __name__=='__main__':main()
