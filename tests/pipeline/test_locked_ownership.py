"""Visible ownership stays fixed through nearest fill, watershed and island cleanup."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.split_parts import _compute_ownership, split_parts

ROOT = Path(__file__).resolve().parents[2]


class TestLockedOwnership(unittest.TestCase):
    def fixture(self):
        height,width=60,60
        src=np.zeros((height,width,4),dtype=np.uint8)
        src[4:56,4:56]=[120,155,175,255]
        pet=np.zeros((height,width),dtype=bool)
        pet[32:46,4:15]=True
        pet[22,22]=True  # Tiny visible pet detail must not be cleaned into a leg.
        tail=np.zeros_like(pet)
        tail[12:50,47:50]=True
        tail[47:50,20:50]=True
        tail[35:50,20:23]=True  # Curves around the lower body.
        subject=(src[...,3]>0)&~pet&~tail
        src[pet]=[160,70,80,255]
        src[tail]=[180,140,60,255]
        seed=subject.copy();seed[24:40]=False
        parts=[
            {'name':'pet','mask':pet,'allowed_mask':pet,'ownership_locked':True,'band':10,'depth':12},
            {'name':'tail','mask':tail,'allowed_mask':tail,'ownership_locked':True,'band':10,'depth':12},
            {'name':'legs','mask':seed,'allowed_mask':subject,'band':10,'depth':12},
        ]
        return src,parts,pet,tail,subject

    def test_adjoining_pet_and_wrapped_tail_are_not_absorbed(self):
        src,parts,pet,tail,subject=self.fixture()
        lab=_compute_ownership(src,parts,{}, {},min_island_px=100)
        self.assertTrue((lab[pet]==0).all())
        self.assertTrue((lab[tail]==1).all())
        self.assertTrue((lab[subject]==2).all())
        self.assertEqual(int(lab[22,22]),0)
        # Even an overbroad later part cannot overwrite a locked earlier owner.
        parts[2]['mask']=src[...,3]>0
        parts[2].pop('allowed_mask')
        broad=_compute_ownership(src,parts,{}, {},min_island_px=100)
        self.assertTrue((broad[pet]==0).all())
        self.assertTrue((broad[tail]==1).all())

    def test_conflicting_locks_and_missing_domain_fail(self):
        src,parts,pet,tail,subject=self.fixture()
        parts[1]['mask']=pet.copy();parts[1]['allowed_mask']=pet.copy()
        with self.assertRaisesRegex(ValueError,'Overlapping locked'):
            _compute_ownership(src,parts,{}, {},12)
        src,parts,pet,tail,subject=self.fixture()
        parts[2]['allowed_mask'][5,5]=False
        with self.assertRaisesRegex(ValueError,'Unassigned foreground'):
            _compute_ownership(src,parts,{}, {},12)

    def test_exact_masks_cannot_silently_resize_or_disappear(self):
        src,parts,*_=self.fixture()
        parts[0]['mask']=np.zeros((12,12),dtype=bool)
        with self.assertRaisesRegex(ValueError,'size mismatch'):
            _compute_ownership(src,parts,{}, {},12)
        parts[0]['mask']='missing-pet-mask.png'
        with self.assertRaisesRegex(ValueError,'missing'):
            _compute_ownership(src,parts,{}, {},12)

    def test_visible_ownership_and_hidden_tail_completion_are_separate(self):
        src,parts,pet,tail,subject=self.fixture()
        work=ROOT/'.studio-work';work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work,prefix='locked-ownership-test-') as td:
            root=Path(td);source=root/'input.png';Image.fromarray(src).save(source)
            # File-backed masks are the production path; keep metadata serializable.
            for part in parts:
                for key in ('mask','allowed_mask'):
                    path=root/(part['name']+'-'+key+'.png')
                    Image.fromarray(part[key].astype(np.uint8)*255).save(path)
                    part[key]=str(path)
                part.update(kind='static' if part['name']=='pet' else 'soft',pivot=[30,30],guides=[(None,None,None,'smooth')])
            result=split_parts(source,parts,root/'parts',seams={('pet','legs'):None,('tail','legs'):None},min_island_px=100)
            labels=np.load(root/'parts/work/labels.npy')
            self.assertTrue((labels[pet]==0).all())
            self.assertTrue((labels[tail]==1).all())
            import json
            meta=json.loads((root/'parts/parts.json').read_text())
            rest=json.loads((root/'parts/rest_diff.json').read_text())
            self.assertEqual(rest['maxRGBA'],0)
            tail_hidden=0
            for part in meta['parts']:
                image=np.asarray(Image.open(root/'parts'/part['image']).convert('RGBA'))
                alpha=image[...,3]
                x,y,w,h=part['bbox']
                if part['name']=='tail':
                    hidden=(alpha>0)&subject[y:y+h,x:x+w]
                    tail_hidden=int(hidden.sum())
                    # Tail completion uses tail gold, never copied human blue.
                    self.assertTrue((image[hidden,0]>image[hidden,2]).all())
                if part['name']=='legs':
                    # Top layer may not borrow visible pet/tail pixels via seam feathering.
                    self.assertFalse(((alpha>0)&(pet|tail)[y:y+h,x:x+w]).any())
            self.assertGreater(tail_hidden,0,'tail root needs completion behind the body')


if __name__ == '__main__':
    unittest.main()
