'use strict';
const http = require('node:http');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const readline = require('node:readline');
const PROTOCOL = 'portrait-vision-v1';
const MAX_IMAGE = 10 * 1024 * 1024;
const MAX_BODY = Math.ceil(MAX_IMAGE * 4 / 3) + 4096;
const TERMINAL = new Set(['completed','failed','outcome_unknown']);
const error = (code,message,status=400) => Object.assign(new Error(message),{code,status});

function validateInput(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw error('INVALID_REQUEST','需要 JSON 对象。');
  const allowed=new Set(['jobId','inputSha256','width','height','imageBase64','preset']);
  if(Object.keys(body).some(k=>!allowed.has(k)))throw error('INVALID_REQUEST','含未支持字段。');
  if(!/^[A-Za-z0-9_-]{1,64}$/.test(body.jobId||''))throw error('INVALID_JOB_ID','jobId 无效。');
  if(!/^[a-f0-9]{64}$/.test(body.inputSha256||''))throw error('INVALID_SHA256','图片摘要无效。');
  if(![body.width,body.height].every(n=>Number.isInteger(n)&&n>0&&n<=8192))throw error('INVALID_SIZE','图片尺寸无效。');
  const raw=body.imageBase64;
  if(typeof raw!=='string'||raw.length>MAX_BODY||raw.length%4||!/^[A-Za-z0-9+/]+={0,2}$/.test(raw))throw error('INVALID_IMAGE','需要完整 PNG base64 字节。');
  const bytes=Buffer.from(raw,'base64');
  if(bytes.length>MAX_IMAGE||bytes.toString('base64')!==raw)throw error('INVALID_IMAGE','图片无效或超过 10 MiB。');
  if(bytes.length<33||!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))||bytes.toString('ascii',12,16)!=='IHDR'||bytes.readUInt32BE(8)!==13)throw error('INVALID_IMAGE','只接受预处理 PNG，不接受路径或 URL。');
  if(bytes.readUInt32BE(16)!==body.width||bytes.readUInt32BE(20)!==body.height)throw error('INPUT_SIZE_MISMATCH','PNG 尺寸不匹配。');
  if(crypto.createHash('sha256').update(bytes).digest('hex')!==body.inputSha256)throw error('INPUT_HASH_MISMATCH','图片摘要不匹配。');
  if(body.preset!==undefined&&!['agile_full','heavy_weapon_full','bust_half'].includes(body.preset))throw error('INVALID_PRESET','预设无效。');
  return {...body,preset:body.preset||'agile_full'};
}

function createBridge({notify=()=>{},now=()=>Date.now(),timeoutMs=900000}={}) {
  let root,dataDir,server,baseUrl,token,instanceId,binding,timer;
  const jobs=new Map(),images=new Map();let tail=Promise.resolve();
  const serial=fn=>{const p=tail.then(fn);tail=p.catch(()=>{});return p;};
  const clone=obj=>JSON.parse(JSON.stringify(obj));
  async function writeJSON(file,data){const tmp=file+'.'+crypto.randomBytes(8).toString('hex')+'.new';await fs.writeFile(tmp,JSON.stringify(data,null,2)+'\n',{mode:0o600,flag:'wx'});await fs.rename(tmp,file);}
  const save=job=>writeJSON(path.join(dataDir,'jobs',job.jobId+'.json'),job);
  const status=()=>({protocol:PROTOCOL,configured:!!root,listening:!!server?.listening,projectRoot:root,
    connectionFile:dataDir&&path.join(dataDir,'connection.json'),baseUrl,instanceId,taskBinding:binding,
    createRequestKey:root&&('pv-create-'+crypto.createHash('sha256').update(root).digest('hex').slice(0,32)),
    jobs:[...jobs.values()].map(j=>({jobId:j.jobId,status:j.status,taskId:j.taskId,runId:j.runId}))});
  async function configure(projectRoot){
    if(typeof projectRoot!=='string'||!path.isAbsolute(projectRoot))throw error('INVALID_PROJECT','需要已有项目绝对路径。');
    const realRoot=await fs.realpath(projectRoot);
    if(root&&root!==realRoot)throw error('PROJECT_ALREADY_BOUND','插件已绑定另一个项目，请先停止连接。');
    if(root)return status();
    await fs.access(path.join(realRoot,'src','studio','server.py'));
    let actual=realRoot;
    for(const component of ['.studio-work','vision-bridge']) {
      const candidate=path.join(actual,component);
      try{await fs.mkdir(candidate,{mode:0o700});}catch(e){if(e.code!=='EEXIST')throw e;}
      actual=await fs.realpath(candidate);
      if(!actual.startsWith(realRoot+path.sep))throw error('INVALID_PROJECT','连接目录不能指向项目外。');
    }
    root=realRoot;dataDir=actual;await fs.chmod(dataDir,0o700);
    await fs.mkdir(path.join(dataDir,'jobs'),{recursive:true,mode:0o700});
    if(!(await fs.realpath(path.join(dataDir,'jobs'))).startsWith(dataDir+path.sep))throw error('INVALID_PROJECT','作业目录不能是外部链接。');
    for(const file of await fs.readdir(path.join(dataDir,'jobs'))){
      if(!/^[A-Za-z0-9_-]{1,64}\.json$/.test(file))continue;
      const job=JSON.parse(await fs.readFile(path.join(dataDir,'jobs',file),'utf8'));
      if(!TERMINAL.has(job.status)){job.status='outcome_unknown';job.error={code:'BRIDGE_RESTARTED',message:'连接重启，原输入可能已执行；请检查 Cindy 原任务，不自动重派。',retryable:false};await save(job);}
      jobs.set(job.jobId,job);
    }
    try{binding=JSON.parse(await fs.readFile(path.join(dataDir,'task.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
    return status();
  }
  async function bindTask(b){
    if(!root||!b||typeof b.taskId!=='string'||!b.taskId)throw error('INVALID_TASK_BINDING','缺少有效任务回执。');
    if(binding&&binding.taskId!==b.taskId)throw error('TASK_BINDING_CONFLICT','不能自动替换分析任务。');
    binding=b;await writeJSON(path.join(dataDir,'task.json'),binding);return status();
  }
  async function submit(raw){
    const i=validateInput(raw),old=jobs.get(i.jobId);
    if(old){if(old.inputSha256!==i.inputSha256||old.width!==i.width||old.height!==i.height||old.preset!==i.preset)throw error('JOB_ID_CONFLICT','同一 jobId 已绑定另一输入。',409);return {code:200,job:clone(old)};}
    if(!binding)throw error('BRIDGE_NOT_READY','尚未建立 Cindy 分析任务。',503);
    if([...jobs.values()].some(j=>!TERMINAL.has(j.status)||j.status==='outcome_unknown'))throw error('BRIDGE_BUSY','上一分析尚未完成或结果未知；请检查原任务。',409);
    const job={jobId:i.jobId,inputSha256:i.inputSha256,width:i.width,height:i.height,preset:i.preset,status:'queued',source:'actual_cindy_agent',
      taskId:binding.taskId,requestKey:'portrait-'+i.jobId+'-'+i.inputSha256.slice(0,16),createdAt:now(),deadlineAt:now()+timeoutMs,instanceId};
    jobs.set(job.jobId,job);images.set(job.jobId,i.imageBase64);await save(job);notify('vision-job',{jobId:job.jobId});return {code:202,job:clone(job)};
  }
  async function claim(jobId){const j=jobs.get(jobId);if(!j||j.status!=='queued'||!images.has(jobId))return {claimed:false};j.status='running';j.phase='preparing';await save(j);return {claimed:true,job:clone(j),imageBase64:images.get(jobId),taskBinding:binding};}
  async function update({jobId,patch}){
    const j=jobs.get(jobId);if(!j)throw error('JOB_NOT_FOUND','找不到原作业。',404);
    if(j.status==='completed'||j.status==='failed')return clone(j);
    if(!patch||Object.keys(patch).some(k=>!['status','phase','runId','executionReceipt','result','error'].includes(k)))throw error('INVALID_UPDATE','状态字段无效。');
    if(patch.runId&&j.runId&&patch.runId!==j.runId)throw error('RUN_BINDING_CONFLICT','不能换绑执行。');
    if(patch.status&&!['running','completed','failed','outcome_unknown'].includes(patch.status))throw error('INVALID_UPDATE','状态无效。');
    if(patch.status==='completed'){
      const r=patch.result,e=patch.executionReceipt;
      if(!r||r.input_sha256!==j.inputSha256||JSON.stringify(r.image_size)!==JSON.stringify([j.width,j.height])||!e||e.taskId!==j.taskId||e.runId!==(j.runId||patch.runId)||!e.outputMessageId)throw error('RESULT_BINDING_MISMATCH','结果或执行回执与输入不一致。');
    }
    Object.assign(j,patch,{updatedAt:now()});await save(j);if(TERMINAL.has(j.status))images.delete(jobId);return clone(j);
  }
  async function expire(){for(const j of jobs.values())if(!TERMINAL.has(j.status)&&now()>=j.deadlineAt)await update({jobId:j.jobId,patch:{status:'outcome_unknown',error:{code:'ANALYSIS_TIMEOUT',message:'等待超时，Cindy 原任务可能仍在执行；不自动重提。',retryable:false}}});}
  async function listen(){
    if(!binding)throw error('BRIDGE_NOT_READY','尚未建立 Cindy 分析任务。');if(server?.listening)return status();
    token=crypto.randomBytes(32).toString('hex');instanceId=crypto.randomUUID();
    server=http.createServer(async(req,res)=>{
      const reply=(code,obj)=>{if(res.headersSent)return;res.writeHead(code,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(obj));};
      try{
        if(req.headers.origin||req.headers.host!==new URL(baseUrl).host)throw error('LOCAL_CLIENT_REQUIRED','只接受本机 Python 客户端。',403);
        const got=Buffer.from(req.headers.authorization||''),want=Buffer.from('Bearer '+token);
        if(got.length!==want.length||!crypto.timingSafeEqual(got,want))throw error('UNAUTHORIZED','连接认证失败。',401);
        const url=new URL(req.url,baseUrl);
        if(req.method==='GET'&&url.pathname==='/v1/health'){reply(200,{ok:true,protocol:PROTOCOL,instanceId,taskId:binding.taskId});return;}
        if(req.method==='GET'&&/^\/v1\/jobs\/[A-Za-z0-9_-]{1,64}$/.test(url.pathname)){const j=jobs.get(url.pathname.split('/').pop());if(!j)throw error('JOB_NOT_FOUND','找不到该输入。',404);reply(200,clone(j));return;}
        if(req.method!=='POST'||url.pathname!=='/v1/jobs')throw error('NOT_FOUND','未知接口。',404);
        if(!String(req.headers['content-type']||'').startsWith('application/json'))throw error('INVALID_CONTENT_TYPE','使用 application/json。',415);
        const chunks=[];let size=0;for await(const c of req){size+=c.length;if(size>MAX_BODY)throw error('REQUEST_TOO_LARGE','图片过大。',413);chunks.push(c);}
        let body;try{body=JSON.parse(Buffer.concat(chunks).toString('utf8'));}catch{throw error('INVALID_JSON','JSON 无效。');}
        const out=await serial(()=>submit(body));reply(out.code,out.job);
      }catch(e){reply(e.status||500,{error:{code:e.code||'BRIDGE_ERROR',message:e.message,retryable:false}});}
    });
    server.requestTimeout=30000;
    await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
    baseUrl='http://127.0.0.1:'+server.address().port;
    await writeJSON(path.join(dataDir,'connection.json'),{schemaVersion:1,protocol:PROTOCOL,baseUrl,token,instanceId});
    timer=setInterval(()=>serial(expire).catch(()=>{}),1000);timer.unref();return status();
  }
  async function close(){clearInterval(timer);if(server)await new Promise(resolve=>server.close(resolve));}
  async function request(method,p={}){
    if(method==='bridge/wait'){await new Promise(r=>setTimeout(r,Math.min(5000,Math.max(100,p.ms||1500))));return {};}
    if(method==='bridge/status')return status();
    return serial(async()=>{
      if(method==='bridge/configure')return configure(p.projectRoot);
      if(method==='bridge/bind-task')return bindTask(p);
      if(method==='bridge/listen')return listen();
      if(method==='bridge/claim')return claim(p.jobId);
      if(method==='bridge/update')return update(p);
      throw error('METHOD_NOT_FOUND','未知桥接方法。');
    });
  }
  return {request,close,expire:()=>serial(expire)};
}
if(require.main===module){
  const send=m=>process.stdout.write(JSON.stringify(m)+'\n');
  const bridge=createBridge({notify:(method,params)=>send({jsonrpc:'2.0',method,params})});
  const input=readline.createInterface({input:process.stdin});
  input.on('line',async line=>{let r;try{r=JSON.parse(line);send({jsonrpc:'2.0',id:r.id,result:await bridge.request(r.method,r.params)});}catch(e){send({jsonrpc:'2.0',id:r?.id??null,error:{code:-32000,message:e.message,data:{code:e.code||'BRIDGE_ERROR'}}});}});
  input.on('close',()=>bridge.close().finally(()=>process.exit(0)));
}
module.exports={createBridge,validateInput,PROTOCOL};
