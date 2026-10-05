#!/usr/bin/env python3
"""Independent raw-score audit of the frozen 2x2 policy replay; no project imports."""
import argparse,collections,csv,hashlib,itertools,json,math
from pathlib import Path
import numpy as np

POLICIES=('fixed_forced','fixed_selective','adaptive_forced','adaptive_selective')
MENUS=('independent_pool','same_category_pool')
REWARD=(1.,.8,.6,.4,.2)
PLAN_SHA='4a455bb6bc3d46eb7b2a547130b69008d67ed41f3391e97f50dfd3e87ae7d195'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ensure(v,m):
 if not v:raise AssertionError(m)
def probabilities(vals):
 m=max(vals); es=[math.exp(v-m) for v in vals];den=math.fsum(es);return [v/den for v in es]
def choice_round(probs,theta,fixed,policy):
 ensure(len(probs)==5 and fixed in range(1,6),'trajectory dimensions')
 crossings=[] if theta is None else [r for r,p in enumerate(probs,1) if p>=theta]
 first=crossings[0] if crossings else None
 if policy=='fixed_forced':return fixed
 if policy=='fixed_selective':return fixed if theta is not None and probs[fixed-1]>=theta else None
 if policy=='adaptive_forced':return first if first is not None else 5
 if policy=='adaptive_selective':return first
 raise AssertionError('unknown policy')
def edge_checks():
 tests=[([.4,.6,.8,.2,.9],.6,2,'fixed_selective',2),([.4,.6,.8,.2,.9],.6,2,'adaptive_selective',2),
  ([.4,.6,.8,.2,.9],.6,2,'adaptive_forced',2),([.4]*5,.6,2,'fixed_selective',None),
  ([.4]*5,.6,2,'adaptive_forced',5),([.4]*5,.6,2,'adaptive_selective',None),
  ([.9]*5,None,2,'adaptive_selective',None),([.9]*5,None,2,'adaptive_forced',5),
  ([.9]*5,None,2,'fixed_selective',None),([.9]*5,None,2,'fixed_forced',2),
  ([.8,.1,.9,.2,.9],.6,2,'adaptive_selective',1),([.8,.1,.9,.2,.9],.6,2,'fixed_selective',None),
  ([.4,.4,.4,.4,.7],.6,1,'adaptive_selective',5)]
 for ps,th,fi,po,ex in tests:ensure(choice_round(ps,th,fi,po)==ex,'edge '+po)
 return len(tests)
def interval(x,probs=(.025,.975)):
 finite=x[np.isfinite(x)]
 return np.quantile(finite,probs).tolist() if len(finite) else None

def reconstruct(root,plan_path):
 ensure(sha(plan_path)==PLAN_SHA,'frozen plan changed')
 manifest=read(root/'artifact_manifest.json')
 for r in manifest['files']:
  p=root/r['path'];ensure(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'input manifest '+r['path'])
 package=read(root/'run/pilot.json');jobs={j['score_id']:j for j in package['jobs']};gold={q['qid']:q for q in read(root/'inputs/main_dataset.json')['questions']}
 fits={f['condition']:f for f in read(root/'inputs/fitted_parameters.json')['parameters'] if f['answer_source']=='plain'}
 fitted_hash=sha(root/'inputs/fitted_parameters.json')
 ensure(fitted_hash==package['frozen_policy']['source_fitted_parameters_sha256'],'frozen fit hash')
 raw=[json.loads(line) for line in (root/'run/output/qwen7b/scores.jsonl').read_text().splitlines()]
 ensure(len(raw)==len(jobs)==8000 and len({r['score_id'] for r in raw})==8000,'score coverage')
 ensure(sha(root/'run/output/qwen7b/scores.jsonl')==read(root/'run/output/qwen7b/receipt.json')['scores_sha256'],'scores receipt')
 traj=collections.defaultdict(dict);max_softmax_discrepancy=0.
 for r in raw:
  j=jobs[r['score_id']];ensure(all(r[k]==v for k,v in j.items() if k!='prompt'),'score/public identity')
  if j['arm']!='plain':continue
  ls=[r['raw_action_logits'][l] for l in 'ABCD'];ensure(all(math.isfinite(v) for v in ls),'finite logits')
  ps=probabilities(ls); candidate='ABCD'[max(range(4),key=lambda i:ls[i])];canonical=j['option_source_ids'][candidate]
  max_softmax_discrepancy=max(max_softmax_discrepancy,max(abs(p-r['conditional_answer_probabilities'][l]) for p,l in zip(ps,'ABCD')))
  menu=next(m for m in gold[j['qid']]['menus'] if m['condition']==j['condition']);f=fits[j['condition']]
  conf=max(ps);clip=min(f['feature_clip'][1],max(f['feature_clip'][0],conf));z=f['intercept']+f['slope']*math.log(clip/(1-clip));cal=1/(1+math.exp(-z))
  t=traj[j['qid'],j['condition'],j['rotation']];ensure(j['round'] not in t,'duplicated trajectory round')
  t[j['round']]={'raw_confidence':conf,'calibrated_probability':cal,'choice':canonical,'correct':canonical==menu['gold_option_id'],'gold':menu['gold_option_id'],'source_score_id':r['score_id'],'prefix_id':j['prefix_id']}
 ensure(max_softmax_discrepancy<=2e-6,'softmax discrepancy')
 qids=sorted({k[0] for k in traj});ensure(len(qids)==100 and len(traj)==800,'independent units')
 episodes={};qrows=collections.defaultdict(list)
 for (qid,menu,rotation),t in sorted(traj.items()):
  ensure(set(t)==set(range(1,6)),'five rounds missing');f=fits[menu];fixed=int(f['selected_fixed_policy'].split('_')[1]);p=[t[i]['calibrated_probability'] for i in range(1,6)]
  for policy in POLICIES:
   n=choice_round(p,f['selected_threshold'],fixed,policy);committed=n is not None;correct=committed and t[n]['correct'];reward=(REWARD[n-1] if correct else -1.) if committed else 0.
   e={'qid':qid,'condition':menu,'rotation':rotation,'policy':policy,'round':n,'committed':committed,'correct':correct,'wrong':committed and not correct,'terminal_pass':not committed,'canonical_choice':t[n]['choice'] if committed else None,'reward':reward,'observed_round':n if committed else 5}
   episodes[qid,menu,rotation,policy]=e;qrows[qid,menu,policy].append(e)
 # Directly prove old cells are unchanged episode by episode.
 old=list(csv.DictReader((root/'analysis/episodes.csv').open()));matched=0
 for r in old:
  if r['policy'] not in ('frozen_plain_threshold','frozen_plain_fixed'):continue
  policy='adaptive_selective' if r['policy']=='frozen_plain_threshold' else 'fixed_forced';e=episodes[r['qid'],r['condition'],int(r['rotation']),policy]
  for k in ('round','committed','correct','wrong','terminal_pass','canonical_choice','reward','observed_round'):
   ensure(r[k]==('' if e[k] is None else str(e[k])),'old episode mismatch '+k)
  matched+=1
 ensure(matched==1600,'old comparison coverage')
 means={k:{field:math.fsum(float(r[field]) for r in rs)/4 for field in ('reward','committed','correct','wrong','terminal_pass','observed_round')} for k,rs in qrows.items()}
 for rs in qrows.values():ensure(sorted(r['rotation'] for r in rs)==[0,1,2,3],'rotation balance')
 idx=np.random.default_rng(1).integers(0,100,(20000,100));arrays={};summaries=[]
 for menu,policy in itertools.product(MENUS,POLICIES):
  vs={field:np.array([means[q,menu,policy][field] for q in qids]) for field in ('reward','committed','correct','wrong','terminal_pass','observed_round')};arrays[menu,policy]=vs
  denom=vs['committed'][idx].sum(1);risk=np.divide(vs['wrong'][idx].sum(1),denom,out=np.full(20000,np.nan),where=denom>0)
  summary={'condition':menu,'policy':policy,'mean_reward':float(vs['reward'].mean()),'reward_ci95':interval(vs['reward'][idx].mean(1)),
   'coverage':float(vs['committed'].mean()),'coverage_ci95':interval(vs['committed'][idx].mean(1)),
   'conditional_error':float(vs['wrong'].sum()/vs['committed'].sum()) if vs['committed'].sum() else None,'conditional_error_ci95':interval(risk),
   'committed_count':int(sum(r['committed'] for r in episodes.values() if r['condition']==menu and r['policy']==policy)),
   'correct_count':int(sum(r['correct'] for r in episodes.values() if r['condition']==menu and r['policy']==policy))}
  summaries.append(summary)
 contrasts=[]
 for menu in MENUS:
  for name,coefs in [('adaptive_minus_fixed_selective',{'adaptive_selective':1,'fixed_selective':-1}),
   ('adaptive_minus_fixed_forced',{'adaptive_forced':1,'fixed_forced':-1}),
   ('selective_minus_forced_fixed',{'fixed_selective':1,'fixed_forced':-1}),
   ('selective_minus_forced_adaptive',{'adaptive_selective':1,'adaptive_forced':-1}),
   ('interaction',{'adaptive_selective':1,'fixed_selective':-1,'adaptive_forced':-1,'fixed_forced':1})]:
   delta=sum(coef*arrays[menu,policy]['reward'] for policy,coef in coefs.items());boot=delta[idx].mean(1)
   contrasts.append({'condition':menu,'contrast':name,'mean_delta':float(delta.mean()),'ci95':interval(boot),'ci97_5':interval(boot,(.0125,.9875)) if name=='adaptive_minus_fixed_selective' else None})
 examples=[]
 for menu in MENUS:
  for category in ('adaptive_benefit','adaptive_harm','no_crossing'):
   possible=[]
   for qid in qids:
    for rot in range(4):
     fs=episodes[qid,menu,rot,'fixed_selective'];as_=episodes[qid,menu,rot,'adaptive_selective'];diff=as_['reward']-fs['reward']
     if (category=='adaptive_benefit' and diff>0) or (category=='adaptive_harm' and diff<0) or (category=='no_crossing' and not as_['committed']):possible.append((qid,rot))
   if not possible:continue
   qid,rot=possible[0];question=gold[qid];menu_obj=next(m for m in question['menus'] if m['condition']==menu)
   states=[]
   for rd,s in sorted(traj[qid,menu,rot].items()):
    prefix=next(p['text'] for p in question['prefixes'] if p['prefix_id']==s['prefix_id']);states.append({'round':rd,**s,'prefix':prefix})
   examples.append({'category':category,'selection_rule':'first lexicographic qid then rotation satisfying the specified category; illustrative not representative',
    'qid':qid,'condition':menu,'rotation':rot,'threshold':fits[menu]['selected_threshold'],'frozen_fixed_round':int(fits[menu]['selected_fixed_policy'].split('_')[1]),
    'question':question['question'],'answer':question['answer'],'options':menu_obj['options'],'gold_option_id':menu_obj['gold_option_id'],'states':states,
    'policies':{pol:episodes[qid,menu,rot,pol] for pol in POLICIES}})
 ensure(fitted_hash==sha(root/'inputs/fitted_parameters.json'),'fit changed')
 return {'status':'passed','manifest_members_verified':len(manifest['files']),'plan_sha256':PLAN_SHA,'raw_rows_validated':8000,'plain_states_reconstructed':4000,
  'episodes_reconstructed':len(episodes),'question_cells_reconstructed':len(means),'n_questions':100,'old_unchanged_episodes_verified':matched,
  'max_softmax_discrepancy':max_softmax_discrepancy,'edge_tests_passed':edge_checks(),'fitting_performed':False,'bootstrap_samples':20000,'bootstrap_seed':1,
  'policy_summaries':summaries,'contrasts':contrasts,'examples':examples},episodes,means,arrays,idx,traj

def compare(result,episodes,means,arrays,idx,traj,analysis):
 rows=list(csv.DictReader((analysis/'episodes.csv').open()));seen=set()
 for r in rows:
  k=(r['qid'],r['condition'],int(r['rotation']),r['policy']);ensure(k not in seen and k in episodes,'main episode coverage');seen.add(k)
  for name in ('round','committed','correct','wrong','terminal_pass','canonical_choice','reward','observed_round'):
   v=episodes[k][name];ensure(r[name]==('' if v is None else str(v)),'main episode mismatch '+name)
 ensure(set(episodes)==seen,'main missing episodes')
 summary=read(analysis/'summary.json');lookup={(s['condition'],s['policy']):s for s in result['policy_summaries']}
 for s in summary['policy_summaries']:
  our=lookup[s['condition'],s['policy']]
  for met,intervalkey in [('mean_reward','reward_ci95'),('coverage','coverage_ci95'),('conditional_error','conditional_error_ci95')]:
   ensure((s[met]['mean'] is None and our[met] is None) or abs(s[met]['mean']-our[met])<1e-12,'main summary '+met)
   ensure((s[met]['ci95'] is None and our[intervalkey] is None) or np.max(np.abs(np.array(s[met]['ci95'])-our[intervalkey]))<1e-12,'main CI '+met)
  for k in ('committed_count','correct_count'):ensure(s[k]==our[k],'main counts')
  ensure(s['wrong_count']==our['committed_count']-our['correct_count'] and s['terminal_pass_count']==400-our['committed_count'],'main wrong/PASS counts')
  values=arrays[s['condition'],s['policy']]['observed_round'];ensure(abs(s['mean_observed_round']['mean']-values.mean())<1e-12,'main observed round')
  ensure(np.max(np.abs(np.array(s['mean_observed_round']['ci95'])-interval(values[idx].mean(1))))<1e-12,'main observed round interval')
 for s in summary['primary_contrasts']:
  our=next(c for c in result['contrasts'] if c['condition']==s['condition'] and c['contrast']=='adaptive_minus_fixed_selective')
  ensure(abs(s['mean_delta']-our['mean_delta'])<1e-12,'main primary point')
  for k in ('ci95','ci97_5'):ensure(np.max(np.abs(np.array(s[k])-our[k]))<1e-12,'main primary interval')
 names={'timing_forced':'adaptive_minus_fixed_forced','timing_selective':'adaptive_minus_fixed_selective',
  'abstention_fixed':'selective_minus_forced_fixed','abstention_adaptive':'selective_minus_forced_adaptive','timing_by_abstention_interaction':'interaction'}
 for s in summary['simple_effects']:
  our=next(c for c in result['contrasts'] if c['condition']==s['condition'] and c['contrast']==names[s['contrast']])
  ensure(abs(s['mean_delta']-our['mean_delta'])<1e-12,'main secondary point')
  ensure(np.max(np.abs(np.array(s['ci95'])-our['ci95']))<1e-12,'main secondary interval')
 def point_boot(menu,policy,metric):
  v=arrays[menu,policy]
  if metric=='coverage':return float(v['committed'].mean()),v['committed'][idx].mean(1)
  if metric=='conditional_error':
   den=v['committed'][idx].sum(1)
   return (float(v['wrong'].sum()/v['committed'].sum()) if v['committed'].sum() else None,
    np.divide(v['wrong'][idx].sum(1),den,out=np.full(20000,np.nan),where=den>0))
  raise AssertionError('unexpected secondary metric')
 for s in summary['metric_deltas']:
  left,lb=point_boot(s['condition'],s['left'],s['metric']);right,rb=point_boot(s['condition'],s['right'],s['metric']);boot=lb-rb
  ensure(abs(s['mean_delta']-(left-right))<1e-12,'main paired coverage/error point')
  ensure(np.max(np.abs(np.array(s['ci95'])-interval(boot)))<1e-12,'main paired coverage/error CI')
  ensure(s['defined_resamples']==int(np.isfinite(boot).sum()) and s['total_resamples']==20000,'main secondary resamples')
 pq=list(csv.DictReader((analysis/'per_question.csv').open()));seen=set()
 for r in pq:
  k=(r['qid'],r['condition'],r['policy']);ensure(k not in seen and k in means,'main question coverage');seen.add(k)
  ensure(all(abs(float(r[f])-v)<1e-12 for f,v in means[k].items()),'main question fields')
 ensure(seen==set(means),'main question missing')
 traces=list(csv.DictReader((analysis/'trajectories.csv').open()));seen=set()
 for r in traces:
  k=(r['qid'],r['condition'],int(r['rotation']));rd=int(r['round']);ensure((k,rd) not in seen,'duplicate main trace');seen.add((k,rd));v=traj[k][rd]
  ensure(r['score_id']==v['source_score_id'] and r['candidate_choice']==v['choice'] and r['candidate_correct']==str(v['correct']),'main trace candidate')
  ensure(abs(float(r['candidate_confidence'])-v['raw_confidence'])<1e-12 and abs(float(r['calibrated_probability'])-v['calibrated_probability'])<1e-12,'main trace confidence')
  ensure(r['threshold_met']==str(v['calibrated_probability']>=float(r['threshold'])),'main trace threshold')
 ensure(len(seen)==4000,'main trace count')
 result['main_comparison']={'episodes_checked':len(rows),'policy_summaries_checked':len(summary['policy_summaries']),'primary_contrasts_checked':len(summary['primary_contrasts']),
  'secondary_reward_contrasts_checked':len(summary['simple_effects']),'paired_coverage_and_error_differences_checked':len(summary['metric_deltas']),
  'per_question_records_checked':len(pq),'trajectory_states_checked':len(traces),
  'summary_sha256':sha(analysis/'summary.json'),'episodes_sha256':sha(analysis/'episodes.csv'),'status':'passed'}

def example_output(result):
 """Return three deterministic, explicitly outcome-selected mechanical examples."""
 selected=[]
 reasons={'adaptive_benefit':'Fixed round 2 confidence is below 0.85, so fixed-selective never answers. Adaptive confidence first exceeds 0.85 at round 4 with the gold candidate, earning +0.4.',
  'adaptive_harm':'Fixed round 2 confidence is below 0.85, so fixed-selective never answers. Adaptive confidence first exceeds 0.85 at round 3 with a wrong candidate, earning -1.',
  'no_crossing':'Confidence never reaches 0.85. Both selective policies terminal-PASS. Adaptive-forced answers the gold candidate at round 5 and earns +0.2; abstention can sacrifice a correct answer.'}
 for category in ('adaptive_benefit','adaptive_harm','no_crossing'):
  x=next(e for e in result['examples'] if e['category']==category and e['condition']=='same_category_pool')
  selected.append({'category':category,'condition':x['condition'],'qid':x['qid'],'rotation':x['rotation'],'full_question':x['question'],
   'early_prefix':x['states'][0]['prefix'],'options':x['options'],'gold_canonical_id':x['gold_option_id'],'gold_answer':x['answer']['accepted'],
   'threshold':x['threshold'],'frozen_fixed_round':x['frozen_fixed_round'],
   'rounds':[{'round':r['round'],'choice':r['choice'],'calibrated_probability':r['calibrated_probability'],'raw_confidence':r['raw_confidence'],'correct':r['correct'],'prefix':r['prefix'],'source_score_id':r['source_score_id']} for r in x['states']],
   'policies':{k:{'round':v['round'],'reward':v['reward'],'choice':v['canonical_choice']} for k,v in x['policies'].items()},
   'why_selected':reasons[category],'selection_rule':x['selection_rule'],
   'mechanical_checks':{'raw_logits_softmax_recomputed':True,'candidate_identity_restored_from_display_rotation':True,'evaluator_gold_join_verified':True,'frozen_calibration_recomputed':True,'threshold_crossings_and_rewards_verified':True},
   'caveat':'Illustrative outcome-selected saved-score trajectory, not a representative sample or a human factual audit of the question. Candidate IDs in this record are canonical, not displayed labels.'})
 return {'schema_version':'imcqa-fixed-abstention-checked-trajectories-v1','evidence_scope':'Mechanically checked illustrative outcome-selected examples','examples':selected}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--analysis',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--examples-out',type=Path);a=p.parse_args()
 result,ep,means,arrays,idx,traj=reconstruct(a.root,a.plan)
 if a.analysis:compare(result,ep,means,arrays,idx,traj,a.analysis)
 if a.examples_out:
  a.examples_out.parent.mkdir(parents=True,exist_ok=True);a.examples_out.write_text(json.dumps(example_output(result),indent=2)+'\n')
 result['auditor_sha256']=sha(__file__);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('examples',)},indent=2))
if __name__=='__main__':main()
