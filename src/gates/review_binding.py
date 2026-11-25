"""Bind explicit visual acceptance to current files; no Host identity is fabricated."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image


class ExportApprovalError(ValueError):
    code = "EXPORT_APPROVAL_REQUIRED"
    retryable = False


VISUAL_DIMENSIONS = ("semanticCompleteness", "rigidShape", "softDeformation", "contactsOcclusion", "loopAppearance")


def resolve_evidence_path(value: Union[str, Path], *, root: Union[str, Path],
                          base: Union[str, Path, None] = None) -> Path:
    """Resolve before reading; absolute in-task paths are valid, escapes are not."""
    boundary=Path(root).resolve()
    if not isinstance(value,(str,Path)) or not str(value).strip():
        raise ExportApprovalError('验收证据路径缺失。')
    path=Path(value)
    if '..' in path.parts:
        raise ExportApprovalError('验收证据路径不能包含父目录跳转。')
    candidate=path if path.is_absolute() else Path(base or boundary)/path
    try:
        resolved=candidate.resolve(strict=True)
        resolved.relative_to(boundary)
    except (OSError,ValueError,RuntimeError) as exc:
        raise ExportApprovalError('验收证据缺失或越过当前任务证据根。') from exc
    if not resolved.is_file():raise ExportApprovalError('验收证据不是文件。')
    return resolved


def read_evidence_mask(value, expected_sha, size, *, root, base=None):
    """Hash and decode the same bounded PNG bytes, never silently resize."""
    import numpy as np
    path=resolve_evidence_path(value,root=root,base=base)
    with path.open('rb') as stream:raw=stream.read(12*1024*1024+1)
    actual=hashlib.sha256(raw).hexdigest()
    if not expected_sha or actual!=expected_sha:
        raise ExportApprovalError('验收证据掩码摘要变化，请重新审计与复核。')
    if len(raw)>12*1024*1024 or not raw.startswith(bytes([137,80,78,71,13,10,26,10])):
        raise ExportApprovalError('验收证据必须是有界PNG。')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format!='PNG' or image.mode not in ('L','1') or image.size!=tuple(size):
                raise ExportApprovalError('验收证据格式或尺寸与当前画布不一致。')
            array=np.asarray(image.convert('L'))
            if not np.isin(array,[0,255]).all():raise ExportApprovalError('验收证据不是二值掩码。')
    except (OSError,ValueError) as exc:
        raise ExportApprovalError('无法读取当前验收证据PNG。') from exc
    return path,actual,array>0


def read_evidence_rgba(value, expected_sha, size, *, root, base=None):
    import numpy as np
    path=resolve_evidence_path(value,root=root,base=base)
    with path.open('rb') as stream:raw=stream.read(24*1024*1024+1)
    sha=hashlib.sha256(raw).hexdigest()
    if len(raw)>24*1024*1024 or not expected_sha or sha!=expected_sha:
        raise ExportApprovalError('Canvas材质字节或摘要变化。')
    with Image.open(io.BytesIO(raw)) as image:
        if image.format!='PNG' or image.mode!='RGBA' or image.size!=tuple(size):
            raise ExportApprovalError('Canvas材质格式/尺寸变化。')
        array=np.asarray(image,dtype=np.uint8).copy()
    return path,sha,array


def _bind_contract_evidence(rdir, build, normalized, source_alpha=None):
    """Bind and recheck external masks inside the caller's job, not a report path."""
    import numpy as np
    from src.studio.vision_schema import rasterize_polygons
    contract=build['moduleContract'];size=tuple(normalized['image_size'])
    if tuple(contract.get('canvas',()))!=size:raise ExportApprovalError('验收画布不匹配。')
    root=rdir.parent;hashes={};visible={};hidden={}
    def read(value,sha):
        path,actual,mask=read_evidence_mask(value,sha,size,root=root,base=root)
        hashes['task-evidence/'+path.relative_to(root).as_posix()]=actual
        return mask
    for module in contract.get('modules',[]):
        names=module.get('pieces') or [];paths=module.get('visibleMasks') or []
        if len(names)!=len(paths):raise ExportApprovalError('可见证据清单不完整。')
        for name,path in zip(names,paths):
            visible[name]=read(path,contract.get('visibleMaskSHA256',{}).get(name))
    occlusions=contract.get('occlusions') or []
    provided_backs={'piece_'+oc['back'] for oc in normalized['occlusions'] if oc['completion']=='provided'}
    for oc in occlusions:
        if oc.get('hiddenMask'):
            back=oc['back'];mask=read(oc['hiddenMask'],contract.get('hiddenMaskSHA256',{}).get(back))
            hidden[back]=hidden.get(back,np.zeros(mask.shape,bool))|mask
        if oc.get('providedImage'):
            if oc['back'] not in provided_backs:raise ExportApprovalError('Canvas材质没有原合同声明。')
            path,sha,_=read_evidence_rgba(oc['providedImage'],oc.get('providedImageSHA256'),size,root=root,base=root)
            hashes['task-evidence/'+path.relative_to(root).as_posix()]=sha
    source_pieces={'piece_'+p['id']:p for m in normalized['modules'] for p in m['pieces']}
    source_pairs={('piece_'+o['front'],'piece_'+o['back']):o for o in normalized['occlusions']}
    seen=set()
    expected_provided={}
    for oc in occlusions:
        if oc.get('autoSeamUnderlap'):continue
        pair=(oc.get('front'),oc.get('back'))
        if pair not in source_pairs or pair in seen:raise ExportApprovalError('遮挡证据与原合同引用不一致。')
        seen.add(pair);front,back=pair;original=source_pairs[pair]
        if oc.get('completionMode')!=original['completion']:raise ExportApprovalError('补全类型与原合同不一致。')
        if front not in visible or back not in visible:raise ExportApprovalError('遮挡可见证据缺失。')
        expected=rasterize_polygons(original['polygons'],*size)&visible[front]&~visible[back]
        piece=source_pieces[back]
        if not piece.get('mask_png_base64'):
            expected &= rasterize_polygons(piece['polygons'],*size)
        if original['completion']=='provided':
            from src.mivo.canvas_assets import decode_canvas_completion
            supplied=decode_canvas_completion(original['provided'],image_size=size,input_sha256=normalized['input_sha256'])
            if source_alpha is None:raise ExportApprovalError('Canvas材质缺少原图alpha核对。')
            fill_zone=rasterize_polygons(original.get('completion_polygons') or original['polygons'],*size)
            take=expected&fill_zone&(source_alpha==255)&(supplied[...,3]>0)
            expected_provided.setdefault(back,np.zeros_like(supplied))[take]=supplied[take]
        target=read(oc.get('sourceHiddenTargetMask',''),oc.get('sourceHiddenTargetMaskSHA256'))
        remaining=read(oc.get('remainingHiddenMask',''),oc.get('remainingHiddenMaskSHA256'))
        expected_remaining=target&~hidden.get(back,np.zeros(target.shape,bool))
        if (oc.get('sourceHiddenTargetRegistration')!='original-back-claim-and-occlusion-and-visible-front'
                or not np.array_equal(target,expected) or not np.array_equal(remaining,expected_remaining)
                or oc.get('remainingHiddenPixels')!=int(remaining.sum())):
            raise ExportApprovalError('当前遮挡掩码内容与合同目标/剩余域不一致，旧审核失效。')
    if seen!=set(source_pairs):raise ExportApprovalError('原合同遮挡证据未完整登记。')
    for back,expected in expected_provided.items():
        refs=[o for o in occlusions if o['back']==back and o.get('providedImage')]
        if not refs:raise ExportApprovalError('Canvas材质未进入构建证据。')
        for oc in refs:
            _,_,actual=read_evidence_rgba(oc['providedImage'],oc.get('providedImageSHA256'),size,root=root,base=root)
            if not np.array_equal(actual,expected):raise ExportApprovalError('Canvas回接材质与原合同不一致。')
    return hashes


def module_review_binding(job_dir: Union[str, Path], contract: Dict[str, Any]) -> Dict[str, str]:
    """Bind the existing pre-animation confirmation to actual inspection images."""
    from src.studio.module_contract import contract_fingerprint
    directory = Path(job_dir).resolve()
    source_path = resolve_evidence_path('matted.png',root=directory)
    analysis_path = resolve_evidence_path('vision/vision.json',root=directory)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    analysis_sha = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    manifest_path = resolve_evidence_path('review-modules/module-review.json',root=directory)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    contract_sha = contract_fingerprint(contract)
    if (manifest.get('source_sha256') != source_sha or source_sha != contract.get('input_sha256')
            or manifest.get('analysis_sha256') != analysis_sha or manifest.get('contract_sha256') != contract_sha):
        raise ExportApprovalError('模块对照与当前输入或规划不一致，请重新生成对照并检查。')
    hashes = {}
    for module in contract['modules']:
        for suffix in ('-visible.png', '-comparison.jpg'):
            path = resolve_evidence_path(manifest_path.parent / (module['id'] + suffix),root=directory)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {'input_sha256': source_sha, 'analysis_sha256': analysis_sha,
            'contract_sha256': contract_sha, 'inspection_sha256': _digest(hashes)}


def validate_module_review(evidence: Dict[str, Any], binding: Dict[str, str], contract: Dict[str, Any]) -> None:
    if (not isinstance(evidence, dict) or evidence.get('binding') != binding
            or evidence.get('source') not in ('current_cindy_conversation', 'user_ui')
            or not isinstance(evidence.get('reviewer'), str) or not evidence['reviewer'].strip()
            or evidence.get('original_and_modules_viewed') is not True):
        raise ExportApprovalError('生成前请在同一次绑定确认中记录当前原图和完整模块对照的实际观察。')
    observations = evidence.get('modules')
    if not isinstance(observations, dict) or set(observations) != {m['id'] for m in contract['modules']}:
        raise ExportApprovalError('每个完整模块都需要实际观察记录，不能以零漏覆盖替代。')
    for item in observations.values():
        if (not isinstance(item, dict) or item.get('status') != 'pass'
                or not isinstance(item.get('notes'), str) or not item['notes'].strip()):
            raise ExportApprovalError('模块仍有缺失、失败或待审观察，请先修正完整归属再生成动画。')


def validate_visual_evidence(evidence: Dict[str, Any], contract: Dict[str, Any], *, decision: str) -> None:
    """Validate explicit observation scope, not an assertion that software watched the image."""
    observations = evidence.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(VISUAL_DIMENSIONS):
        raise ExportApprovalError("视觉复核需要完整的五项观察：语义边缘、保形、柔性、接触补全、循环观感。")
    soft_present = any(m.get("motion_type") == "soft" for m in contract.get("modules", []))
    contacts_present = bool(contract.get("connections") or contract.get("occlusions"))
    for key in VISUAL_DIMENSIONS:
        item = observations[key]
        if (not isinstance(item, dict) or item.get("status") not in ("pass", "fail", "pending", "not_applicable")
                or not isinstance(item.get("notes"), str) or not item["notes"].strip()):
            raise ExportApprovalError("每项视觉观察都需要状态和具体依据。")
        if item["status"] == "not_applicable" and not (
                key == "softDeformation" and not soft_present
                or key == "contactsOcclusion" and not contacts_present):
            raise ExportApprovalError("适用的视觉项目不能标记为不适用。")
        if decision == "pass" and item["status"] not in ("pass", "not_applicable"):
            raise ExportApprovalError("仍有失败或待审的视觉子项，不能记录整体通过。")
    if decision == "pass" and (evidence.get("original_and_modules_viewed") is not True
                                or evidence.get("full_loop_observed") is not True):
        raise ExportApprovalError("通过前必须实际对照原图与分件图，并检查原速完整循环。")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _actual_review_context(rig_dir: Union[str, Path], *, source_image: Union[str, Path],
                           analysis_file: Union[str, Path], job_id: str):
    """One checked byte snapshot supplies both binding and visual-plan validation."""
    from src.studio.module_contract import contract_fingerprint, validate_module_contract
    from .module_quality import RULES_VERSION
    rdir = Path(rig_dir).resolve()
    root = rdir.parent
    # rglob does not descend directory symlinks. Reject them (and file/broken
    # links) BEFORE is_file filtering or any artifact reads, so a renderer's
    # images subtree cannot silently disappear from the approval digest.
    artifacts=[];directories=[rdir]
    try:
        while directories:
            directory=directories.pop()
            if directory.is_symlink():
                raise ExportApprovalError('rig内不能使用文件或目录符号链接作为验收产物。')
            directory.resolve(strict=True).relative_to(rdir)
            for path in directory.iterdir():
                if path.is_symlink():
                    raise ExportApprovalError('rig内不能使用文件或目录符号链接作为验收产物。')
                path.resolve(strict=True).relative_to(rdir)
                artifacts.append(path)
                if path.is_dir():directories.append(path)
    except (OSError,ValueError) as exc:
        if isinstance(exc,ExportApprovalError):raise
        raise ExportApprovalError('rig目录不完整或解析后越过验收根。') from exc
    artifacts.sort()
    source = resolve_evidence_path(source_image, root=root, base=root)
    plan_path = resolve_evidence_path(analysis_file, root=root, base=root)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if not job_id or plan.get("job_id") != job_id:
        raise ExportApprovalError("模块规划与复核任务身份不匹配。")
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    with Image.open(io.BytesIO(source_bytes)) as image:
        size = image.size
        if any(o.get('completion')=='provided' for o in plan.get('occlusions',[])):
            import numpy as np
            source_alpha=np.asarray(image.convert('RGBA'))[...,3].copy()
        else:source_alpha=None
    normalized = validate_module_contract(plan, input_sha256=source_sha, image_size=size)
    build_path = resolve_evidence_path(rdir / 'rig-build-report.json', root=rdir)
    build = json.loads(build_path.read_text(encoding="utf-8"))
    contract = build.get("moduleContract") or {}
    if (contract.get("sourceVersion") != 2 or contract.get("sourceSHA256") != source_sha
            or contract.get("contractSHA256") != contract_fingerprint(normalized)):
        raise ExportApprovalError("生成骨架与当前原图或完整模块规划不一致。")
    spine = resolve_evidence_path(build.get("spineFile", ""), root=rdir, base=rdir)
    if not spine.is_file() or not spine.name.endswith(".spine.json"):
        raise ExportApprovalError("当前骨架文件不存在。")
    spine_bytes=spine.read_bytes()
    skeleton=json.loads(spine_bytes)
    # Bind the paths the native renderer actually consumes as well as the tree:
    # an absolute skeleton.images or escaping attachment name must not evade
    # the symlink check by referring to an external directory directly.
    image_reference=skeleton.get('skeleton',{}).get('images','./images/')
    if not isinstance(image_reference,str) or '..' in Path(image_reference).parts:
        raise ExportApprovalError('贴图目录引用非法或包含父目录跳转。')
    try:
        image_dir=(rdir/image_reference).resolve(strict=True)
        image_dir.relative_to(rdir)
    except (OSError,ValueError,RuntimeError) as exc:
        raise ExportApprovalError('贴图目录缺失或越过rig验收根。') from exc
    if not image_dir.is_dir():raise ExportApprovalError('贴图引用不是目录。')
    for slot in skeleton.get('slots',[]):
        attachment=slot.get('attachment')
        if not isinstance(attachment,str) or not attachment:
            raise ExportApprovalError('当前绘制slot缺少有效贴图引用。')
        resolve_evidence_path(image_dir/(attachment+'.png'),root=rdir)
    hashes = _bind_contract_evidence(rdir,build,normalized,source_alpha)
    source_relative = source.relative_to(root).as_posix()
    analysis_relative = plan_path.relative_to(root).as_posix()
    analysis_sha = hashlib.sha256(plan_bytes).hexdigest()
    hashes['task-input/source/'+source_relative] = source_sha
    hashes['task-input/analysis/'+analysis_relative] = analysis_sha
    for path in artifacts:
        if path.is_symlink():
            raise ExportApprovalError('rig符号链接已变化，旧验收失效。')
        if not path.is_file() or not (path.name.endswith(".spine.json") or path.suffix.lower() == ".png"
                                      or path.name == "rig-build-report.json"):
            continue
        checked = resolve_evidence_path(path, root=rdir)
        data=spine_bytes if checked==spine else checked.read_bytes()
        hashes[str(path.relative_to(rdir))] = hashlib.sha256(data).hexdigest()
    return {"job_id": job_id, "input_sha256": source_sha,
            "analysis_sha256": analysis_sha,
            "source_relative_path": source_relative, "analysis_relative_path": analysis_relative,
            "artifact_sha256": _digest(hashes), "qc_rules_version": RULES_VERSION}, normalized


def actual_review_binding(rig_dir: Union[str, Path], *, source_image: Union[str, Path],
                         analysis_file: Union[str, Path], job_id: str) -> Dict[str, str]:
    """Bind controlled task-relative input/evidence locations and their bytes."""
    return _actual_review_context(rig_dir, source_image=source_image,
                                  analysis_file=analysis_file, job_id=job_id)[0]


def _ensure_user_signoff(rdir, signoff_path, health_report_path=None):
    """Explicit human acceptance overrides advisory QC, not broken artifacts."""
    try:
        signoff = json.loads(Path(signoff_path).read_text(encoding="utf-8"))
        if (signoff.get("decision") != "pass" or not any(
                isinstance(signoff.get(key), str) and signoff[key].strip()
                for key in ("user_quote", "reviewer"))):
            raise ExportApprovalError("人工签收必须明确 pass 并有用户原话或验收人。")
        build = json.loads((rdir / "rig-build-report.json").read_text(encoding="utf-8"))
        spine = json.loads((rdir / build["spineFile"]).read_text(encoding="utf-8"))
        images = rdir / "images"
        required = {slot["attachment"] for slot in spine["slots"]}
        if not required or not spine.get("bones"):
            raise ExportApprovalError("骨架或贴图清单为空。")
        for name in required:
            with Image.open(images / (name + ".png")) as image:
                image.verify()
        report_path = Path(health_report_path) if health_report_path else rdir / "health-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if report_path.exists():
            rest = report.get("hardGates", {}).get("rest_diff")
            if not isinstance(rest, dict) or rest.get("passed") is not True:
                raise ExportApprovalError("rest_diff 缺失或失败，不能以人工签收覆盖基础结构错误。")
            checks = report.get("quality", {}).get("bindingMotion", {}).get("checks", {})
            rigid_checks = [checks.get("rigidity"), checks.get("rigid"),
                            report.get("structuralChecks", {}).get("weapon_rigidity")]
            for check in rigid_checks:
                if isinstance(check, dict) and (check.get("status") == "fail" or check.get("passed") is False):
                    raise ExportApprovalError("刚体结构明确失败，不能以人工签收覆盖。")
        # Preserve all machine findings. Human release is a separate decision.
        return {**report, "userSignoff": signoff, "exportAuthorization": {
            "basis": "explicit_user_signoff", "exportAllowed": True,
            "signoffPath": str(Path(signoff_path).resolve()),
            "materialMode": (build.get("moduleContract") or {}).get("materialMode", "source_sliced")}}
    except ExportApprovalError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ExportApprovalError("人工签收或骨架/贴图/基础检查文件缺失、不可读。") from exc


def ensure_export_approved(rig_dir: Union[str, Path], *,
                           health_report_path: Union[str, Path, None] = None,
                           user_signoff_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Public exporter gate; a stored boolean without an exact explicit review is insufficient."""
    from .module_quality import recompute_quality, RULES_VERSION
    rdir = Path(rig_dir).resolve()
    if user_signoff_path is not None:
        return _ensure_user_signoff(rdir, user_signoff_path, health_report_path)
    try:
        build_path = rdir / 'rig-build-report.json'
        if build_path.exists():
            build_path = resolve_evidence_path(build_path, root=rdir)
            build = json.loads(build_path.read_text(encoding='utf-8'))
            if (build.get('moduleContract') or {}).get('materialMode') == 'canvas_layers':
                raise ExportApprovalError('Canvas独立层绑定候选尚未接入完整质量与视觉验收，不能正式导出。')
        report_path = resolve_evidence_path(
            health_report_path if health_report_path is not None else rdir / "health-report.json",
            root=rdir.parent, base=rdir.parent)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        q = report.get("quality") or {}
        if q.get("rulesVersion") != RULES_VERSION:
            raise ExportApprovalError("QC规则版本已变化，请按当前规则重新检查并复核，旧确认不能导出。")
        if q.get("contractApplicable") is not True:
            raise ExportApprovalError("旧诊断产物没有完整模块 V2 验收，不能导出为合格工程。")
        visual = q.get("visual") or {}
        evidence = visual.get("evidence") or {}
        paths = evidence.get("paths") or {}
        expected = evidence.get("review_binding") or {}
        # Re-read external evidence before trusting serialized audit states.
        # Geometry need not rerun here, but stale/moved/replaced masks must not
        # survive a current binding check or a forged updated file SHA alone.
        if all(paths.get(k) for k in ('source_image','analysis_file')) and expected.get('job_id'):
            current, checked_plan = _actual_review_context(rdir, source_image=paths["source_image"],
                analysis_file=paths["analysis_file"], job_id=expected["job_id"])
            if current != expected:
                raise ExportApprovalError("原图、规划或动画证据文件已变化，旧视觉通过已失效。")
        else:
            raise ExportApprovalError("缺少当前原图、规划和验收证据绑定。")
        # Recompute from component states rather than trusting serialized exportAllowed.
        recompute_quality(report)
        if report["quality"]["overall"].get("exportAllowed") is not True:
            raise ExportApprovalError("完整模块的输入、绑定、机器与视觉项目尚未全部通过。")
        visual = q.get("visual") or {}
        evidence = visual.get("evidence") or {}
        if (evidence.get("source") not in ("user_ui", "current_cindy_conversation")
                or evidence.get("full_loop_observed") is not True
                or not evidence.get("reviewer") or not evidence.get("notes")):
            raise ExportApprovalError("缺少明确的完整循环视觉复核记录。")
        validate_visual_evidence(evidence, checked_plan, decision="pass")
    except ExportApprovalError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ExportApprovalError("验收记录缺失或无法与当前真实文件核对，不能导出。") from exc
    return report
