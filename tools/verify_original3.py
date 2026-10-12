"""Bind new checkpoint selections to exports and completed official-reset episodes."""
import hashlib,json
TASKS={'mixer_operating':4000,'manipulate_pipette':20000,'sample_loading':20000}
SOURCE_SHA={'mixer_operating':'b14c94d6544c3bc4936279988852d632da654ea6ba4728a9ba6f297d8f899802','manipulate_pipette':'48d2b5472352a819df701b2c3433d3aac9e23e700f0d73050c0feeeb97e0e7a8','sample_loading':'467da5b83da0bfdcf0df70564ed9d14da624819942a2b3fb82bf7648e6574131'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,manifest):
 folder=root/'verification/original3_20261012'
 proof=json.loads((folder/'proof.json').read_text())
 assert proof['passed'] and proof['episodes_per_task']==20
 required={'candidate_source_sha256.json','model_manifest.json','platform_status.json','stage_proof.json'}
 for task in TASKS:
  required.update(f'{task}/{name}' for name in ('audit.json','metrics.json','launch_contract.json','cleanup.json','preflight.json','gpu.csv','job.exit','driver.exit','native_fault.txt'))
 assert required.issubset(proof['evidence_sha256']), 'Incomplete new-run evidence manifest'
 for name,sha in proof['evidence_sha256'].items():
  p=folder/name
  assert p.resolve().is_relative_to(folder.resolve()) and digest(p)==sha,name
 tested=json.loads((folder/'candidate_source_sha256.json').read_text())
 from verify_runtime_diagnostics import BUILD_TOOLS
 for name,sha in manifest.items():
  if (name.startswith(('runtime/','policy/','tools/')) and name not in BUILD_TOOLS) or name=='task_mapping.json':assert tested.get(name)==sha,name
 prior=json.loads((root/'verification/pre_original3_task_mapping.json').read_text())
 current=json.loads((root/'task_mapping.json').read_text())
 for key in ('horizon','execute_steps','prefix_steps'):assert prior[key]==current[key]
 for task,entry in prior['tasks'].items():
  if task not in TASKS:assert current['tasks'][task]==entry,task
 model_manifest=json.loads((folder/'model_manifest.json').read_text())
 states=json.loads((folder/'platform_status.json').read_text())
 assert set(states)==set(TASKS) and all(v=='completed' for v in states.values())
 old_models=json.loads((root/'verification/runtime_diagnostics_20261012/handle_basket/launch_contract.json').read_text())['model_manifest']
 for name,record in old_models['files'].items():assert model_manifest['files'][name]==record,name
 accepted={}
 for task,step in TASKS.items():
  artifact='mixer_operating_terminal_press_ft4k' if task=='mixer_operating' else f'{task}_160k_ft{step//1000}k'
  expected={'artifact':artifact,'checkpoint_step':step,'base_checkpoint_step':10000 if task=='mixer_operating' else 160000}
  if task=='mixer_operating':expected['shared_initialization_step']=160000
  assert current['tasks'][task]==expected
  export=json.loads((folder/f'{task}_export.json').read_text())
  assert export['passed'] and export['exported_tensors_exactly_equal_source'] and export['tensor_count']==1135
  assert (export['task'],export['checkpoint_step'])==(task,step)
  assert export['source_checkpoint_sha256']==SOURCE_SHA[task]
  for name,sha in export['evidence_sha256'].items():assert digest(folder/'source_history'/task/name)==sha
  if task=='mixer_operating':
   selection=json.loads((folder/'mixer4k_history/audit.json').read_text())
   assert selection['passed'] and selection['checkpoint_sha256']==SOURCE_SHA[task] and selection['checkpoint_step']==step and selection['successes']==45 and selection['episodes']==50
   for name,sha in selection['evidence_sha256'].items():assert digest(folder/'mixer4k_history'/name)==sha
  else:
   selection=json.loads((folder/'source_history'/task/'historical_EVAL_COMPLETE.json').read_text())
   assert selection['checkpoint_sha256']==SOURCE_SHA[task] and selection['checkpoint_step']==step
  assert export['all_tensors_finite'] and export['normalization_equal_to_shared'] and export['gripper_encoding']=='normalized_1'
  for name,key in [('model.pt','model_sha256'),('config.yaml','config_sha256')]:assert model_manifest['files'][artifact+'/'+name]['sha256']==export[key]
  audit=json.loads((folder/task/'audit.json').read_text());c=json.loads((folder/task/'launch_contract.json').read_text())
  assert audit['passed'] and audit['task']==task and audit['episodes']==20
  assert audit['official_revision']=='8818c82df5fa2522809b8105e7018950dcc9a9a8' and audit['eval_reset_sync_steps']==0
  assert audit['model_manifest']==c['model_manifest']==model_manifest
  assert digest(folder/task/'metrics.json')==audit['metrics_sha256']
  metrics=json.loads((folder/task/'metrics.json').read_text())
  assert len(metrics['episodes'])==20
  assert sum(bool(e['success']) for e in metrics['episodes'])==audit['summary']['success_count']==metrics['summary']['success_count']
  assert audit['model_artifact'].rsplit('/',1)[-1]==artifact and c['artifact'].rsplit('/',1)[-1]==artifact
  assert (folder/task/'job.exit').read_text().strip()==(folder/task/'driver.exit').read_text().strip()=='0'
  assert (folder/task/'native_fault.txt').stat().st_size==0
  assert c['task_mapping_sha256']==manifest['task_mapping.json']
  assert (c['horizon'],c['execute_steps'],c['prefix_steps'],c['policy_seed'])==(30,29,0,20260908)
  assert c['attention_backend']=='sdpa' and c['evaluator_wrapper_sha256']==manifest['tools/run_official_evaluator.py']
  cleanup=json.loads((folder/task/'cleanup.json').read_text());assert cleanup['primary_exit_code']==0 and not cleanup['warnings']
  accepted[task]={'artifact':artifact,'episodes':20,'successes':audit['summary']['success_count'],'model_sha256':export['model_sha256'],'gpu_peak_MiB':audit['gpu_peak_MiB']}
 # Preserve exact N20 evidence for seven unchanged effective policies.
 retained={}
 for task,entry in current['tasks'].items():
  if task in TASKS:continue
  p=folder/'retained7_n20'/task;a=json.loads((p/'audit.json').read_text());c=json.loads((p/'launch_contract.json').read_text())
  assert a['passed'] and a['episodes']==20 and a['task']==task
  assert a['eval_reset_sync_steps']==0 and a['metrics_sha256']==digest(p/'metrics.json')
  assert a['model_artifact'].rsplit('/',1)[-1]==entry['artifact']
  assert (c['horizon'],c['execute_steps'],c['prefix_steps'])==(30,20 if task=='items_handover' else 29,0)
  assert c['task_mapping_sha256']==digest(root/'verification/pre_original3_task_mapping.json')
  for name,record in model_manifest['files'].items():
   if name.startswith(('processor/',entry['artifact']+'/')):assert c['model_manifest']['files'][name]==record
  retained[task]={'episodes':20,'successes':a['summary']['success_count'],'artifact':entry['artifact']}
 return {'new_three':accepted,'retained_seven':retained,'scope':'Three new N20 runs plus seven unchanged effective-policy N20 runs; separate batches'}
