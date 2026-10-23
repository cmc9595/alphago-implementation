"""Independent KataGo analysis of the saved ladder case, for diagnosis only."""
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/diagnostics/ladder_20261018'


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'runs/position_checks/ladder_20261018.json').read_text())
    assert len(source['history'])==73
    moves=[['B' if i%2==0 else 'W',v] for i,v in enumerate(source['history'])]
    model=ROOT/'data/katago/kata1-b18c384nbt-s9281911808-d4126622491.bin.gz'
    config=OUT/'katago_reference.cfg'
    config.write_text('\n'.join([
        'logFile = runs/diagnostics/ladder_20261018/katago_engine.log',
        'maxVisits = 512','numAnalysisThreads = 1','numSearchThreadsPerAnalysisThread = 4',
        'nnMaxBatchSize = 8','nnCacheSizePowerOfTwo = 14','nnMutexPoolSizePowerOfTwo = 12',
        'reportAnalysisWinratesAs = WHITE','rootNumSymmetriesToSample = 1','analysisPVLen = 20',
        'searchRandSeed = 2026101802','wideRootNoise = 0.0',
    ])+'\n')
    base={'moves':moves,'rules':'chinese','komi':source['komi'],'boardXSize':19,'boardYSize':19,
          'maxVisits':512,'includeOwnership':True,'includePolicy':True}
    queries=[{**base,'id':'before_free'}]
    for v in ['H9','M3','M4']:
        queries.append({**base,'id':'force_'+v,'allowMoves':[{'player':'W','moves':[v],'untilDepth':1}]})
        queries.append({**base,'id':'after_'+v,'moves':moves+[['W',v]]})
    ownpath=OUT/'own.json'
    if ownpath.exists():
        own=json.loads(ownpath.read_text())
        for v,entries in own.get('forced',{}).get('new_value',{}).items():
            pv=entries[-1]['candidates'][0]['pv']
            for depth in [4,8]:
                if len(pv)>=depth:
                    extra=[['W' if i%2==0 else 'B',m] for i,m in enumerate(pv[:depth])]
                    queries.append({**base,'id':f'future_{v}_{depth}','moves':moves+extra,'maxVisits':256})
    atomic(OUT/'reference_requests.json',queries)
    command=[str(ROOT/'data/katago/src/cpp/build/katago'),'analysis','-config',str(config),'-model',str(model)]
    result={'engine':'KataGo 1.16.3 OpenCL','network':model.name,'network_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'perspective':'WHITE','rules':'chinese','komi':source['komi'],'queries':{}}
    atomic(OUT/'reference.json',result)
    with (OUT/'katago_stderr.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        responses=queue.Queue()
        def reader():
            for line in proc.stdout:
                responses.put(line)
            responses.put(None)
        threading.Thread(target=reader,daemon=True).start()
        try:
            for query in queries:proc.stdin.write(json.dumps(query)+'\n')
            proc.stdin.flush();proc.stdin.close()
            done=0; deadline=time.monotonic()+600
            while done<len(queries):
                line=responses.get(timeout=max(.01,deadline-time.monotonic()))
                if line is None:raise RuntimeError('KataGo ended early; see katago_stderr.log')
                row=json.loads(line)
                if 'error' in row:raise RuntimeError(str(row))
                if row.get('isDuringSearch',False) or 'rootInfo' not in row:continue
                result['queries'][row['id']]=row;done+=1
                atomic(OUT/'reference.json',result)
                print(json.dumps({'reference_done':done,'total':len(queries),'id':row['id'],
                                  'white_win':row['rootInfo']['winrate'],'white_score_lead':row['rootInfo']['scoreLead'],
                                  'best':row['moveInfos'][0]['move'] if row['moveInfos'] else None}),flush=True)
            proc.wait(timeout=30)
            if proc.returncode:raise RuntimeError('KataGo exit '+str(proc.returncode))
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=10)
    print('REFERENCE DONE',flush=True)


if __name__=='__main__':main()
