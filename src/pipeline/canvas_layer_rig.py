"""Compile Canvas material binding candidates using existing typed rig rules.

This is not Studio intake or visual approval. Preview mode records placement
warnings; formal export requires a separate explicit human signoff.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.gates.review_binding import resolve_evidence_path, read_evidence_rgba
from src.studio.module_contract import (validate_module_contract, MODULE_RULE_VERSION, BODY_ROLES,
                                        validate_hidden_attachment_region)
from src.studio.vision_schema import VisionReviewRequired, rasterize_polygons
from .module_rig import (_bone, _topological, _validate_rig_mask_placement, validate_module_rig_spec)


def compile_canvas_layer_bindings(parts_dir, plan, out_dir, *, evidence_root,
                                  character="canvas_layers", strict_placement=True):
    """Create a new, explicit binding candidate. Never infer or move joint points."""
    import base64
    advisory_notes = {}
    placement_warnings = []
    allowed = {'pose_type', 'anatomy', 'modules', 'connections', 'uncertainties',
               'occlusions', 'input_issues', 'boundary_refinements'}
    if not isinstance(plan, dict) or set(plan) - allowed:
        raise ValueError('Canvas binding plan contains unsupported fields')
    # Strict callers keep their original rejection behavior. Preview callers
    # retain declarations as advisories without executing completion requests.
    if not isinstance(plan.get('occlusions', []), list):
        raise ValueError('occlusions must be a list')
    for key in ('input_issues', 'boundary_refinements'):
        value = plan.get(key, [])
        if not isinstance(value, list):
            raise ValueError(key + ' must be a list')
        if value and not strict_placement:
            advisory_notes[key] = copy.deepcopy(value)
        elif value:
            raise VisionReviewRequired('Canvas绑定候选尚未支持声明 ' + key + '；保留原输入待审，不丢弃后继续。')
    root = Path(evidence_root).resolve()
    manifest_path = resolve_evidence_path(Path(parts_dir) / 'parts.json', root=root)
    pdir = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes(); material = json.loads(manifest_bytes)
    if material.get('materialMode') != 'canvas_layers':
        raise ValueError('Expected independent Canvas layer parts')
    w, h = material['canvas']
    original = resolve_evidence_path(material['originalSource'], root=root)
    if hashlib.sha256(original.read_bytes()).hexdigest() != material['originalSourceSHA256']:
        raise ValueError('Original source changed after materialization')
    setup, _, _ = read_evidence_rgba(material['source'], material['sourceSHA256'], (w, h), root=root)
    if strict_placement and plan.get('pose_type') != 'airborne':
        raise VisionReviewRequired('Canvas绑定候选当前仅验证浮空主链；站立足锚须完成专门接入，不自动降级。')
    if any((strict_placement and c.get('type') == 'foot_anchor') or
           (c.get('hidden_by') and c.get('type') != 'attachment')
           for c in plan.get('connections', [])):
        raise VisionReviewRequired('Canvas绑定候选不假冒足锚或隐藏关节验证。')
    modules = copy.deepcopy(plan.get('modules', []))
    by_id = {m['id']: m for m in modules}
    if len(by_id) != len(modules):
        raise ValueError('Duplicate Canvas module')
    for m in modules:
        if m.get('pieces'):
            raise ValueError('Layer pieces come from the material manifest, not replacement masks')
        m['pieces'] = []
    arrays, paths, hashes, alphas, layer_hashes = {}, {}, {}, {}, {}
    module_masks = {mid: np.zeros((h, w), bool) for mid in by_id}
    for part in material['parts']:
        mid, name = part['moduleId'], part['name']
        if mid not in by_id or name in arrays:
            raise ValueError('Missing module or duplicate layer piece')
        m = by_id[mid]
        grounded_foot = (not strict_placement and plan.get('pose_type') == 'grounded'
                         and m.get('role') == 'foot' and m.get('motion_type') == 'static'
                         and part['kind'] == 'rigid')
        if m.get('parent') != part['parent'] or (m.get('motion_type') != part['kind'] and not grounded_foot):
            raise ValueError('Binding cannot change material module parent or motion kind')
        if not np.allclose(np.asarray(m['pivot']) * [w, h], part['pivot'], atol=1e-5, rtol=0):
            raise ValueError('Binding cannot silently move the declared material pivot')
        layer = part['canvasLayer']
        _, _, rgba = read_evidence_rgba(layer['image'], layer['sha256'], (w, h), root=root)
        cropped_path = resolve_evidence_path(part['image'], root=pdir, base=pdir)
        data = cropped_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != part['imageSHA256']:
            raise ValueError('Materialized part changed')
        x, y, pw, ph = part['bbox']
        if (any(type(v) is not int for v in (x, y, pw, ph)) or min(x, y) < 0
                or min(pw, ph) <= 0 or x+pw > w or y+ph > h):
            raise ValueError('Invalid part bbox')
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != 'RGBA' or image.size != (pw, ph):
                raise ValueError('Material crop format/size changed')
            if not np.array_equal(np.asarray(image), rgba[y:y+ph, x:x+pw]):
                raise ValueError('Material crop differs from registered Canvas layer')
        mask = rgba[..., 3] > 0
        outside = mask.copy(); outside[y:y+ph, x:x+pw] = False
        if outside.any():
            raise ValueError('Material crop omits nontransparent layer pixels')
        arrays[name] = mask; module_masks[mid] |= mask
        alphas[name] = rgba[..., 3].copy(); layer_hashes[name] = layer['sha256']
        buf = io.BytesIO(); Image.fromarray(mask.astype('uint8')*255).save(buf, format='PNG')
        raw = buf.getvalue(); hashes[name] = hashlib.sha256(raw).hexdigest()
        m['pieces'].append({'id': name, 'draw_order': part['drawOrder'],
                            'mask_png_base64': base64.b64encode(raw).decode(), 'mask_sha256': hashes[name]})
    declared_occlusions = copy.deepcopy(plan.get('occlusions', []))
    if not strict_placement and declared_occlusions:
        advisory_notes['occlusions'] = copy.deepcopy(declared_occlusions)
        # Layers already contain their material. Do not execute legacy completion
        # requests here; retain them verbatim as unresolved preview advisories.
        for oc in declared_occlusions:
            if oc.get('completion', 'none') != 'none':
                oc['completion'] = 'none'
                for key in ('provided', 'completion_polygons', 'derive_band'):
                    oc.pop(key, None)
    contract = validate_module_contract({
        'version': 2, 'input_sha256': material['sourceSHA256'], 'image_size': [w, h],
        'coord_mode': 'normalized', 'pose_type': plan.get('pose_type'), 'anatomy': plan.get('anatomy'),
        'modules': modules, 'connections': copy.deepcopy(plan.get('connections', [])),
        'occlusions': declared_occlusions, 'uncertainties': plan.get('uncertainties', []),
    }, input_sha256=material['sourceSHA256'], image_size=(w, h), material_mode='canvas_layers')
    modules = _topological(contract['modules'])
    body = np.zeros((h, w), bool)
    for m in modules:
        if m['role'] in BODY_ROLES: body |= module_masks[m['id']]
    ys = np.nonzero(body)[0]; body_h = float(ys.max()-ys.min()+1) if len(ys) else float(h)
    referenced = {(ref['front'], ref['back']) for c in contract['connections']
                  for ref in c.get('hidden_by', [])}
    target_arrays, occlusion_rows = {}, []
    for oc in contract['occlusions']:
        pair = (oc['front'], oc['back'])
        if not strict_placement and pair not in referenced:
            continue
        if pair not in referenced or oc['completion'] != 'none':
            raise VisionReviewRequired('Canvas occlusions当前仅支持隐藏附件引用的none声明，不能丢弃其他补全语义。')
        front, back = pair
        zone = rasterize_polygons(oc['polygons'], w, h)
        missing = zone & ~arrays[back]
        if not missing.any() or (missing & (alphas[front] == 0)).any():
            if strict_placement:
                raise VisionReviewRequired('显式隐藏附件域有静止未遮区域，不能裁掉缺口后授权。')
            placement_warnings.append({'occlusion': list(pair), 'message': '显式隐藏附件域有静止未遮区域，预览待人眼验收。'})
        target_arrays[pair] = missing
        occlusion_rows.append({'front': front, 'back': back, 'completionMode': 'none',
            'polygons': [[[x*w, y*h] for x, y in polygon] for polygon in oc['polygons']],
            'frontLayerSHA256': layer_hashes[front], 'backLayerSHA256': layer_hashes[back],
            'sourceHiddenTargetRegistration': 'canvas-parent-absence-and-declared-foreground',
            'hiddenAttachment': True, 'sourceInputSHA256': material['sourceSHA256']})
    attachment_support = {}
    for c in contract['connections']:
        if not c.get('hidden_by'): continue
        target = [target_arrays[(ref['front'], ref['back'])] for ref in c['hidden_by']]
        try:
            attachment_support[c['id']] = validate_hidden_attachment_region(
                module_masks[c['a']], target, [c['point'][0]*w, c['point'][1]*h])
        except VisionReviewRequired as exc:
            if strict_placement: raise
            placement_warnings.append({'connection': c['id'], 'message': str(exc)})
    if strict_placement:
        try:
            _validate_rig_mask_placement(modules, contract['connections'], module_masks, w, h, body_h,
                                         hidden_attachment_support=attachment_support)
        except VisionReviewRequired as exc:
            from src.studio.connection_diagnostics import inspect_canvas_connections
            exc.code = 'CONNECTION_REVIEW_REQUIRED'
            exc.details = {'connection_diagnostics': inspect_canvas_connections(pdir, plan, evidence_root=root),
                           'next_action': 'canvas_handoff workflow inspect; source-based planning before repair'}
            raise
    else:
        # Check modules and connections separately so all warnings survive,
        # while original pivots, parent relationships and motion stay intact.
        for module in modules:
            try:
                _validate_rig_mask_placement([module], [], module_masks, w, h, body_h)
            except VisionReviewRequired as exc:
                placement_warnings.append({'module': module['id'], 'message': str(exc)})
        for connection in contract['connections']:
            try:
                _validate_rig_mask_placement([], [connection], module_masks, w, h, body_h,
                                             hidden_attachment_support=attachment_support)
            except VisionReviewRequired as exc:
                placement_warnings.append({'connection': connection['id'], 'message': str(exc)})
    out = Path(out_dir).absolute()
    out.resolve().relative_to(root)
    if out.exists() or out.is_symlink(): raise ValueError('Use a new binding directory')
    out.mkdir(parents=True)
    for name, mask in arrays.items():
        path = out / (name + '-support.png'); Image.fromarray(mask.astype('uint8')*255).save(path)
        paths[name] = str(path)
    for index, oc in enumerate(occlusion_rows):
        path = out / ('hidden-attachment-'+str(index)+'.png')
        Image.fromarray(target_arrays[(oc['front'], oc['back'])].astype('uint8')*255).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('sourceHiddenTargetMask', 'remainingHiddenMask'):
            oc[key] = str(path); oc[key+'SHA256'] = digest
    def point(p): return [round(p[0]*w, 5), round(p[1]*h, 5)]
    bones = [{'name': 'root', 'pos': [w/2., float(h)], 'role': 'root'}]
    chains, skin, projection = {}, {}, []
    for m in modules:
        mid, driver = m['id'], _bone(m['id']); pos = point(m['pivot']); soft = []
        bones.append({'name': driver, 'parent': _bone(m['parent']) if m['parent'] else 'root',
                      'pos': pos, 'role': m['role']})
        if m['motion_type'] == 'soft':
            cname = 'bend_'+mid; pts = [point(p) for p in m['centerline']]
            chains[cname] = {'parent': driver, 'pts': pts, 'count': len(pts)-1}
            soft = [cname+str(i+1) for i in range(len(pts)-1)]
        names = [p['id'] for p in m['pieces']]
        for name in names:
            item = ({'type': 'mesh', 'recipe': 'chain', 'chain': 'bend_'+mid,
                     'cell': max(8, min(24, round(body_h/80)))} if soft else {'type': 'region', 'bone': driver})
            skin[name] = {**item, 'module_id': mid, 'piece_id': name, 'semantic_role': m['role']}
        row = {'id': mid, 'role': m['role'], 'shape': 'soft' if m['motion_type'] == 'soft' else 'rigid', 'parent': m['parent'],
               'bone': driver, 'pivot': pos, 'pieces': names, 'visibleMasks': [paths[n] for n in names],
               'maskSemantics': 'independent_layer_support_not_exclusive_source_ownership',
               'motion': 'static' if m['motion_type'] == 'static' else 'required', 'motionProfile': m['motion'], 'motionType': m['motion_type'],
               'softBones': soft, 'motion_scale': m.get('motion_scale', 1.), 'confidence': m['confidence'], 'side': m.get('side')}
        if m.get('static_reason'): row['staticReason'] = m['static_reason']
        if soft: row['centerline'] = [point(p) for p in m['centerline']]
        if m.get('locked_root'): row['lockedRoot'] = True
        projection.append(row)
    contacts, ik = [], []
    by_bone = {b["name"]: b for b in bones}
    for c in contract['connections']:
        contact = {'id': c['id'], 'type': c['type'], **{side: {
            'module': c[side], 'piece': next(m for m in projection if m['id']==c[side])['pieces'][0],
            'point': point(c['point'])} for side in ('a', 'b')}}
        if c['type'] == 'foot_anchor':
            upper, lower = c['chain']; b = c['b']
            up, low, target = by_bone[_bone(upper)], by_bone[_bone(lower)], by_bone[_bone(b)]
            up['aim'], low['aim'] = low['pos'], target['pos']
            if min(math.dist(up['pos'], low['pos']), math.dist(low['pos'], target['pos'])) < 1.0:
                raise VisionReviewRequired('刚体足锚肢段退化，不能生成足锚。')
            ik.append({'name': 'contact_'+c['id'], 'bones': [_bone(upper), _bone(lower)], 'target': _bone(b)})
            contact.update(type='foot', world=point(c['point']), chain=[_bone(upper), _bone(lower)])
            contact['a'] = {'module': lower, 'piece': next(m for m in projection if m['id'] == lower)['pieces'][0],
                            'point': point(c['point'])}
        if c.get('hidden_by'): contact['hiddenBy'] = copy.deepcopy(c['hidden_by'])
        contacts.append(contact)
    projected = {'sourceVersion': 2, 'version': 1, 'ruleVersion': MODULE_RULE_VERSION,
        'materialMode': 'canvas_layers', 'productionIntakeReady': False, 'exportAllowed': False,
        'sourceSHA256': material['sourceSHA256'], 'originalSourceSHA256': material['originalSourceSHA256'],
        'canvas': [w, h], 'modules': projection, 'requiredModules': list(by_id), 'contacts': contacts,
        'occlusions': occlusion_rows, 'visibleMaskSHA256': hashes,
        'partsJsonSHA256': hashlib.sha256(manifest_bytes).hexdigest(),
        'semanticFidelity': 'pending_original_art_and_full_loop_review',
        'advisoryNotes': advisory_notes}
    rig = {'character': character, 'duration': 8., 'fps': 30, 'key_step': .125,
           'origin': [w/2., float(h)], 'body_height_px': body_h, 'pose_type': contract['pose_type'],
           'bones': bones, 'chains': chains, 'skin': skin, 'ik': ik,
           'placementWarnings': placement_warnings,
           'moduleContract': projected, 'preserve_visual_joints': True}
    validate_module_rig_spec(rig)
    (out/'rig-spec.json').write_text(json.dumps(rig, ensure_ascii=False, indent=2))
    return rig
