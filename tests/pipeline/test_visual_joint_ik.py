"""Two-bone uniform stretch preserves visual setup points and matches both runtimes."""
import copy
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np
from src.render.spine_runtime import Skeleton

ROOT=Path(__file__).resolve().parents[2]


def skeleton():
    return {'bones':[
        {'name':'root'}, {'name':'hips','parent':'root'},
        {'name':'upper','parent':'hips','length':100},
        {'name':'lower','parent':'upper','x':100,'length':100},
        {'name':'foot','parent':'lower','x':100},
        {'name':'target','parent':'root','x':200},
    ],'ik':[{'name':'foot_ik','bones':['upper','lower'],'target':'target','stretch':True,'uniform':True}],
        'slots':[],'skins':[{'name':'default','attachments':{}}],
        'animations':{'idle':{'bones':{'hips':{'translate':[{'x':0,'y':0},{'time':4,'x':-4,'y':4},{'time':8,'x':0,'y':0}]}}}}}


class TestVisualJointIK(unittest.TestCase):
    def test_uniform_stretch_matches_js_and_resets_every_frame(self):
        data=skeleton();times=np.linspace(0,8,33).tolist();sk=Skeleton(data)
        expected=[]
        for t in times:
            sk.apply('idle',t)
            lower=sk.by_name['lower'];target=sk.by_name['target']
            self.assertLess(float(np.linalg.norm(lower.tip_world()-[target.worldX,target.worldY])),1e-8)
            expected.append([[b.worldX,b.worldY,b.a,b.b,b.c,b.d] for b in sk.bones])
        self.assertEqual(sk.by_name['upper'].scaleX,1.0)
        np.testing.assert_allclose(expected[0],expected[-1],atol=1e-10)
        self.assertGreater(np.ptp(np.asarray(expected)[:,1,0]),3.9)
        script = "import fs from 'node:fs';import {Spine42Skeleton} from './src/player/spine42-runtime.js';const p=JSON.parse(fs.readFileSync(0,'utf8'));const s=new Spine42Skeleton(p.skeleton);console.log(JSON.stringify(p.times.map(t=>{s.apply('idle',t);return s.bones.map(b=>[b.worldX,b.worldY,b.a,b.b,b.c,b.d]);})));"
        run=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps({'skeleton':data,'times':times}),text=True,capture_output=True,cwd=ROOT,check=True)
        np.testing.assert_allclose(np.asarray(json.loads(run.stdout)),expected,rtol=1e-10,atol=1e-9)

    def test_excess_stretch_and_minimum_reach_fail_explicitly(self):
        data=skeleton();data['bones'][-1]['x']=213
        with self.assertRaisesRegex(ValueError,'exceeds 1.06'):
            Skeleton(data).apply('idle',0)
        data=skeleton();data['bones'][3]['length']=40;data['bones'][-1]['x']=20
        with self.assertRaisesRegex(ValueError,'minimum reach'):
            Skeleton(data).apply('idle',0)
        # Legacy constraints without stretch retain their previous clamped behavior.
        data=skeleton();data['ik'][0].pop('stretch');data['ik'][0].pop('uniform');data['bones'][-1]['x']=213
        sk=Skeleton(data);sk.apply('idle',0)
        self.assertEqual(sk.by_name['upper'].scaleX,1)
        self.assertGreater(abs(sk.by_name['lower'].tip_world()[0]-213),12)


if __name__=='__main__':unittest.main()
