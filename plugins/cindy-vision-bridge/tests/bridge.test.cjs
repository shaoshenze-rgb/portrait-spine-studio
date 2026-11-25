'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const crypto=require('node:crypto');
const {createBridge}=require('../node/worker.cjs');
const {createController}=require('../main.js');
const WORK=path.resolve(__dirname,'../../../.studio-work/vision-bridge-tests');
const PNG='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB9kAAAAASUVORK5CYII=';
function input(id='one'){return {jobId:id,inputSha256:crypto.createHash('sha256').update(Buffer.from(PNG,'base64')).digest('hex'),width:1,height:1,imageBase64:PNG};}
const CONFIG={agentKind:'codex',providerId:'cindy',model:'vision-test',effort:'high'};
async function fixture(t,options={}){
  await fs.mkdir(WORK,{recursive:true});const root=await fs.mkdtemp(path.join(WORK,'case-'));
  await fs.mkdir(path.join(root,'src/studio'),{recursive:true});await fs.writeFile(path.join(root,'src/studio/server.py'),'# fixture');
  const events=[];const bridge=createBridge({...options,notify:(method,params)=>events.push({method,params})});
  t.after(()=>bridge.close());
  await bridge.request('bridge/configure',{projectRoot:root});
  return {root,bridge,events};
}
async function serving(t,options){const f=await fixture(t,options);await f.bridge.request('bridge/bind-task',{taskId:'task-1',resolvedConfig:CONFIG});await f.bridge.request('bridge/listen');f.conn=JSON.parse(await fs.readFile(path.join(f.root,'.studio-work/vision-bridge/connection.json'),'utf8'));return f;}
async function http(f,route,body,extra={}){const r=await fetch(f.conn.baseUrl+route,{method:body?'POST':'GET',headers:{Authorization:'Bearer '+f.conn.token,'Content-Type':'application/json',...extra},...(body?{body:JSON.stringify(body)}:{})});return {status:r.status,body:await r.json()};}
function apiFor(bridge,overrides={}){
  const calls={create:[],send:[]};let body;
  const task={taskId:'task-1',revision:1,resolvedConfig:CONFIG,permissionMode:'ask'};
  const tasks={
    capabilities:async()=>({operations:['create','get','send','getRun','readMessages'],sourceCallContext:true,createPermissionMode:true,messageAttachments:{kinds:['image'],maxBytes:20*1024*1024,maxItemBytes:20*1024*1024}}),
    create:async args=>{calls.create.push(args);return task;},get:async()=>task,
    send:async args=>{calls.send.push(args);return {runId:'run-1'};},
    getRun:async()=>({taskId:'task-1',runId:'run-1',status:'completed',outputMessageId:'output-1',acceptedConfig:CONFIG,execution:{instanceId:'actual-test',generation:1}}),
    readMessages:async()=>({messages:[{id:'unrelated',role:'assistant',text:'not our result'},{id:'output-1',role:'assistant',text:JSON.stringify(body)}]}),...overrides
  };
  return {api:{tasks,node:{request:async({method,params})=>({ok:true,result:await bridge.request(method,params)})},send:async()=>{}},calls,setResult:r=>{body=r;}};
}
function resultFor(i){return {version:1,input_sha256:i.inputSha256,image_size:[i.width,i.height],coord_mode:'normalized',pose_type:'airborne',joints:{head_top:[.5,.1],neck:[.5,.2],chest:[.5,.4],hips:[.5,.6]},entities:[],uncertainties:['test-only']};}

test('localhost bearer auth, origin refusal, private connection file',async t=>{
  const f=await serving(t);assert.equal((await http(f,'/v1/health')).status,200);
  assert.equal((await http(f,'/v1/health',null,{Authorization:'Bearer wrong'})).status,401);
  assert.equal((await http(f,'/v1/health',null,{Origin:'https://untrusted.example'})).status,403);
  assert.equal((await fs.stat(path.join(f.root,'.studio-work/vision-bridge/connection.json'))).mode&0o777,0o600);
  assert.equal((await f.bridge.request('bridge/status')).token,undefined);
});
test('actual bytes bound to SHA and dimensions; repeated ID dispatches once',async t=>{
  const f=await serving(t),i=input();
  assert.equal((await http(f,'/v1/jobs',{...i,inputSha256:'a'.repeat(64)})).body.error.code,'INPUT_HASH_MISMATCH');
  assert.equal((await http(f,'/v1/jobs',{...i,width:2})).body.error.code,'INPUT_SIZE_MISMATCH');
  assert.equal((await http(f,'/v1/jobs',i)).status,202);assert.equal((await http(f,'/v1/jobs',i)).status,200);
  assert.equal(f.events.length,1);assert.equal((await http(f,'/v1/jobs',{...i,preset:'bust_half'})).status,409);
  assert.equal((await http(f,'/v1/jobs',input('two'))).body.error.code,'BRIDGE_BUSY');
});
test('restart marks in-flight input unknown, same ID never resends',async t=>{
  const f=await serving(t);await http(f,'/v1/jobs',input());await f.bridge.close();
  const replay=[];const again=createBridge({notify:(...args)=>replay.push(args)});t.after(()=>again.close());
  const state=await again.request('bridge/configure',{projectRoot:f.root});assert.equal(state.jobs[0].status,'outcome_unknown');
  await again.request('bridge/listen');f.conn=JSON.parse(await fs.readFile(path.join(f.root,'.studio-work/vision-bridge/connection.json'),'utf8'));
  const r=await http(f,'/v1/jobs',input());assert.equal(r.body.status,'outcome_unknown');assert.equal(replay.length,0);
});
test('deadline becomes unknown and retains original task identity',async t=>{
  let clock=100;const f=await serving(t,{now:()=>clock,timeoutMs:100});await http(f,'/v1/jobs',input());clock=201;await f.bridge.expire();
  const r=await http(f,'/v1/jobs/one');assert.equal(r.body.status,'outcome_unknown');assert.equal(r.body.taskId,'task-1');
});
test('Cindy visible task inherits callId, image attachment reaches exact final JSON',async t=>{
  const f=await fixture(t),stub=apiFor(f.bridge),controller=createController(stub.api),i=input();stub.setResult(resultFor(i));
  const state=await controller.start({projectRoot:f.root},'host-call-123');
  assert.equal(stub.calls.create[0].callId,'host-call-123');assert.equal(stub.calls.create[0].permissionMode,'ask');
  assert.equal(stub.calls.create[0].hidden,undefined);assert.equal(stub.calls.create[0].route,undefined);
  f.conn=JSON.parse(await fs.readFile(state.connectionFile,'utf8'));await http(f,'/v1/jobs',i);await controller.analyze(i.jobId);
  const r=await http(f,'/v1/jobs/one');assert.equal(r.body.status,'completed');assert.equal(r.body.executionReceipt.outputMessageId,'output-1');
  assert.deepEqual(r.body.result,resultFor(i));assert.equal(stub.calls.send.length,1);
  assert.equal(stub.calls.send[0].attachments[0].base64,PNG);assert.equal(stub.calls.send[0].requestKey,r.body.requestKey);
  await controller.analyze(i.jobId);assert.equal(stub.calls.send.length,1);
});
test('missing terminal output ID fails closed without second submission',async t=>{
  const f=await serving(t),stub=apiFor(f.bridge,{getRun:async()=>({status:'completed',runId:'run-1'})}),c=createController(stub.api);
  await http(f,'/v1/jobs',input());await c.analyze('one');const r=await http(f,'/v1/jobs/one');
  assert.equal(r.body.status,'outcome_unknown');assert.equal(r.body.error.code,'FINAL_MESSAGE_UNAVAILABLE');assert.equal(stub.calls.send.length,1);
});
test('unknown send outcome is not retried; mismatching output never completed',async t=>{
  const f=await serving(t);let sends=0;const stub=apiFor(f.bridge,{send:async()=>{sends++;throw new Error('transport closed');}}),c=createController(stub.api);
  await http(f,'/v1/jobs',input());await c.analyze('one');await c.analyze('one');
  assert.equal((await http(f,'/v1/jobs/one')).body.status,'outcome_unknown');assert.equal(sends,1);
});
test('wrong image result rejected with final receipt preserved',async t=>{
  const f=await serving(t),stub=apiFor(f.bridge),c=createController(stub.api);stub.setResult({...resultFor(input()),input_sha256:'b'.repeat(64)});
  await http(f,'/v1/jobs',input());await c.analyze('one');const r=await http(f,'/v1/jobs/one');
  assert.equal(r.body.status,'failed');assert.equal(r.body.error.code,'RESULT_BINDING_MISMATCH');assert.equal(r.body.executionReceipt.outputMessageId,'output-1');
});
test('remote task route is refused before sending any image',async t=>{
  const f=await serving(t),stub=apiFor(f.bridge,{get:async()=>({taskId:'task-1',revision:1,resolvedConfig:{...CONFIG,agentDeviceId:'other-device'}})}),c=createController(stub.api);
  await http(f,'/v1/jobs',input());await c.analyze('one');const r=await http(f,'/v1/jobs/one');
  assert.equal(r.body.status,'failed');assert.equal(r.body.error.code,'REMOTE_ROUTE_NOT_ALLOWED');assert.equal(stub.calls.send.length,0);
});

test('completed without acceptedConfig or native execution stays unknown',async t=>{
  const f=await serving(t),stub=apiFor(f.bridge,{getRun:async()=>({status:'completed',runId:'run-1',outputMessageId:'output-1'})}),c=createController(stub.api);
  stub.setResult(resultFor(input()));await http(f,'/v1/jobs',input());await c.analyze('one');
  const r=await http(f,'/v1/jobs/one');assert.equal(r.body.status,'outcome_unknown');assert.equal(r.body.error.code,'EXECUTION_RECEIPT_UNAVAILABLE');
});
test('notification arriving at previous completion boundary is drained',async t=>{
  const f=await serving(t),stub=apiFor(f.bridge),c=createController(stub.api);stub.setResult(resultFor(input()));
  const original=stub.api.node.request;let inserted=false;
  stub.api.node.request=async args=>{
    const reply=await original(args);
    if(args.method==='bridge/update'&&args.params.patch.status==='completed'&&!inserted){
      inserted=true;assert.equal((await http(f,'/v1/jobs',input('two'))).status,202);await c.analyze('two');
    }
    return reply;
  };
  await http(f,'/v1/jobs',input());await c.analyze('one');
  assert.equal((await http(f,'/v1/jobs/two')).body.status,'completed');assert.equal(stub.calls.send.length,2);
});
