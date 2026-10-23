"""Independent KataGo analysis of the saved ladder case, for diagnosis only."""
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/diagnostics/q5_20261018'


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((OUT/'position.json').read_text())
    assert len(source['history'])==158
    moves=[['B' if i%2==0 else 'W',v] for i,v in enumerate(source['history'])]
    model=ROOT/'data/katago/kata1-b18c384nbt-s9281911808-d4126622491.bin.gz'
    config=OUT/'katago_deeper.cfg'
    config.write_text('\n'.join([
        'logFile = runs/diagnostics/q5_20261018/katago_deeper_engine.log',
        'maxVisits = 512','numAnalysisThreads = 1','numSearchThreadsPerAnalysisThread = 4',
        'nnMaxBatchSize = 8','nnCacheSizePowerOfTwo = 14','nnMutexPoolSizePowerOfTwo = 12',
        'reportAnalysisWinratesAs = WHITE','rootNumSymmetriesToSample = 1','analysisPVLen = 20',
        'wideRootNoise = 0.0',
    ])+'\n')
    base={'moves':moves,'rules':'chinese','komi':source['komi'],'boardXSize':19,'boardYSize':19,
          'maxVisits':512,'includeOwnership':True,'includePolicy':True}
    queries=[]
    for query_id,extra,komi in [
        ('deep_before',[],source['komi']),
        ('deep_force_T7',[],source['komi']),
        ('deep_force_Q13',[],source['komi']),
        ('balanced_after_Q13',[['B','Q13']],-47.5),
        ('balanced_after_Q13_T7',[['B','Q13'],['W','T7']],-47.5),
        ('balanced_after_T7',[['B','T7']],-47.5),
    ]:
        query={**base,'id':query_id,'moves':moves+extra,'komi':komi,'maxVisits':2048}
        if query_id.startswith('deep_force_'):
            query['allowMoves']=[{'player':'B','moves':[query_id.split('_')[-1]],'untilDepth':1}]
        queries.append(query)
    atomic(OUT/'reference_deeper_requests.json',queries)
    command=[str(ROOT/'data/katago/src/cpp/build/katago'),'analysis','-config',str(config),'-model',str(model)]
    result={'engine':'KataGo 1.16.3 OpenCL','network':model.name,'network_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'perspective':'WHITE','rules':'chinese','komi':source['komi'],'queries':{}}
    atomic(OUT/'reference_deeper.json',result)
    with (OUT/'katago_deeper_stderr.log').open('w') as log:
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
                atomic(OUT/'reference_deeper.json',result)
                print(json.dumps({'reference_done':done,'total':len(queries),'id':row['id'],
                                  'white_win':row['rootInfo']['winrate'],'white_score_lead':row['rootInfo']['scoreLead'],
                                  'best':row['moveInfos'][0]['move'] if row['moveInfos'] else None}),flush=True)
            proc.wait(timeout=30)
            if proc.returncode:raise RuntimeError('KataGo exit '+str(proc.returncode))
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=10)
    print('REFERENCE DONE',flush=True)


if __name__=='__main__':main()
