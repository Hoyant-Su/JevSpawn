'use strict';

const $ = (id) => document.getElementById(id);
const colors = {queued: '#46514a', running: '#6edcc6', completed: '#bcf586', failed: '#f49d8c'};
const state = {data: null, events: [], start: 0, end: 0, cursor: 0, playing: false, selected: null, selectedEvent: null, tab: 'output', agents: new Map(), shown: [], count: 0, frame: 0, lastRender: 0, columns: 1, cell: 25, padding: 20, totalAgents: 0, spawnStart: 0, spawnEnd: 0};
const format = (value) => new Intl.NumberFormat('en-US').format(value);
const pretty = (value) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
const elapsed = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
const seconds = (value) => `${new Intl.NumberFormat('en-US', {maximumFractionDigits: 2}).format(value)} s`;

function renderPolicy(summary, metadata) {
  const policy = summary.spawn_policy || 'fixed';
  const copy = {
    fixed: ['FIXED FAN-OUT · TYPED ROLE ROUTING', 'One batch. Every worker visible.', 'One worker per record', 'Task boundaries and worker count are supplied by the workload. The controller selects specialist roles; it does not decide whether to spawn.'],
    single: summary.routing_mode === 'none' ? ['PLAIN GENERATION · NO CONTROLLER', 'One task. One generation.', 'One plain worker per record', 'Each dataset-defined task receives one generation call. This run uses no controller routing, review or repair.'] : ['SINGLE WORKER · WITH ROUTING', 'One task. One routed worker.', 'One worker per record', 'Dataset-defined tasks receive one routed worker each. Additional review and repair workers are disabled by this run policy.'],
    all: ['FORCED REVIEW + REPAIR BASELINE', 'Every task follows the full workflow.', 'Forced review and repair workers', 'Review and repair workers are required by the run policy. Their creation is not an adaptive controller decision.'],
    adaptive: ['ADAPTIVE FAN-OUT · TYPED DECISIONS', 'Decisions create the next workers.', 'Conditional review and repair workers', 'The dataset defines initial tasks. Controller decisions on current drafts gate additional review and repair workers; their recorded parent links identify the source work.'],
  }[policy];
  if (!copy) throw new Error(`Unsupported recorded spawn policy: ${policy}`);
  $('policy-label').textContent = copy[0];
  $('run-headline').textContent = copy[1];
  $('worker-policy').textContent = copy[2];
  $('policy-scope').textContent = copy[3];
  document.title = `JevSpawn · ${copy[0].toLowerCase()}`;
  const generated = summary.controller_mode === 'json' || metadata.controller_mode === 'json';
  $('controller-label').textContent = summary.routing_mode === 'none' ? 'Direct worker execution' : policy === 'fixed' ? generated ? 'Generated JSON role routing' : 'Jev-style structured role routing' : generated ? 'Generated JSON decisions' : 'Typed controller decisions';
  $('spawn-label').textContent = policy === 'fixed' ? 'Routing + worker creation' : 'Recorded spawn interval';
  $('spawn-accounting-values').replaceChildren();
  for (const [key, label] of [['initial_agents', 'Initial role-worker calls'], ['additional_agents', 'Additional role-worker calls'], ['controller_computed_input_tokens', 'Controller tokens computed']]) {
    if (!Number.isFinite(summary[key])) continue;
    const card = document.createElement('div');
    const title = document.createElement('span');
    const value = document.createElement('strong');
    title.textContent = label;
    value.textContent = format(summary[key]);
    card.append(title, value);
    $('spawn-accounting-values').append(card);
  }
  for (const role of ['review', 'repair']) {
    if (!Number.isFinite(summary.roles?.[role])) continue;
    const card = document.createElement('div');
    const title = document.createElement('span');
    const value = document.createElement('strong');
    title.textContent = `${role === 'review' ? 'Review' : 'Repair'} calls`;
    value.textContent = format(summary.roles[role]);
    card.append(title, value);
    $('spawn-accounting-values').append(card);
  }
  $('decision-counts').replaceChildren();
  for (const [stage, choices] of Object.entries(summary.decision_counts || {})) {
    const line = document.createElement('p');
    const label = document.createElement('strong');
    const values = document.createElement('span');
    label.textContent = stage.replaceAll('_', ' ');
    values.textContent = Object.entries(choices).map(([choice, count]) => `${choice}: ${format(count)}`).join(' · ');
    line.append(label, values);
    $('decision-counts').append(line);
  }
  $('spawn-accounting').hidden = !$('spawn-accounting-values').children.length && !$('decision-counts').children.length;
  const architecture = summary.architecture;
  $('execution-design').hidden = !architecture;
  if (architecture) {
    $('architecture-description').textContent = `${architecture.description || architecture.name || 'Summary-declared inference architecture'}. Execution design, not a live cache trace.`;
  }
  const sharedBatches = state.events.filter((event) => event.type === 'batch_completed' && event.role === 'controller' && event.payload.field_mode === 'shared');
  $('shared-compute').hidden = !sharedBatches.length;
  if (sharedBatches.length) {
    const computed = sharedBatches.reduce((total, event) => total + event.payload.computed_input_tokens, 0);
    const logical = sharedBatches.reduce((total, event) => total + event.payload.logical_input_tokens, 0);
    $('shared-compute').textContent = `Run accounting: ${format(sharedBatches.length)} shared-state controller batches processed ${format(computed)} non-padding input tokens, versus ${format(logical)} logical tokens for independent question prefills. These token counts are not measured end-to-end speedups.`;
  }

}

function renderRunMeasurements(metadata) {
  const summary = state.data.summary || {};
  renderPolicy(summary, metadata);
  const profile = summary.profile;
  $('timing-profile').hidden = !profile;
  $('timing-values').replaceChildren();
  $('timing-context').textContent = profile?.interpretation || '';
  $('flow-timing-scope').hidden = !profile?.task_flow_seconds;
  if (profile) {
    if (profile.run_id !== summary.run_id) throw new Error('The timing profile does not match this replay run.');
    const measurements = [['Aggregate throughput', `${profile.tasks_per_second.toFixed(2)} tasks/s`, 'Completed tasks / observed run duration']];
    if (profile.task_flow_seconds) {
      measurements.push(['Task-flow median', seconds(profile.task_flow_seconds.median), `${format(profile.task_flow_seconds.count)} task flows · initial coder start to completion`]);
    }
    for (const role of ['implement', 'review', 'repair']) {
      const latency = profile.worker_latency_seconds[role];
      if (!latency) continue;
      measurements.push([`${role[0].toUpperCase()}${role.slice(1)} call median`, seconds(latency.median), `${format(latency.count)} role calls · batch-result return`]);
    }
    for (const [title, measurement, context] of measurements) {
      const card = document.createElement('div');
      const label = document.createElement('span');
      const value = document.createElement('strong');
      const detail = document.createElement('small');
      label.textContent = title;
      value.textContent = measurement;
      detail.textContent = context;
      card.append(label, value, detail);
      $('timing-values').append(card);
    }
  }
  const comparison = summary.policy_comparison;
  $('policy-comparison').hidden = !comparison;
  $('comparison-rows').replaceChildren();
  $('comparison-context').textContent = comparison?.interpretation || '';
  for (const run of comparison?.policies || []) {
    const row = document.createElement('tr');
    row.classList.toggle('current-run', run.run_id === summary.run_id);
    const policy = document.createElement('th');
    policy.scope = 'row';
    const label = document.createElement('strong');
    const runLabel = document.createElement('small');
    label.textContent = run.policy === 'single' ? run.routing_mode === 'none' ? 'Plain (no controller)' : 'Single with routing' : {adaptive: 'Adaptive', all: 'Forced review + repair'}[run.policy] || run.policy;
    runLabel.textContent = run.run_id;
    policy.append(label, runLabel);
    row.append(policy);
    const fields = [
      `${format(run.passed)} / ${format(run.tasks)} · ${(run.pass_rate * 100).toFixed(2)}%`,
      `${format(run.agents_spawned)} / +${format(run.additional_agents)}`,
      `${(run.splits.humaneval.pass_rate * 100).toFixed(2)}%`,
      `${(run.splits.mbpp_test.pass_rate * 100).toFixed(2)}%`,
      `${format(run.fixed)} / ${format(run.regressed)}`,
      seconds(run.elapsed_seconds),
      format(run.worker_output_tokens),
    ];
    for (const value of fields) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    $('comparison-rows').append(row);
  }
  const job = summary.job || metadata.job;
  $('job-panel').hidden = !job;
  if (job) {
    $('job-id').textContent = job.id;
    $('job-instruction').textContent = job.instruction;
  }
  const measurements = [
    ['spawn-measurement', 'spawn-duration', summary.spawn_elapsed_seconds, seconds],
    ['elapsed-measurement', 'execution-duration', summary.elapsed_seconds, seconds],
    ['peak-measurement', 'measured-peak', summary.peak_running_agents, format],
  ];
  for (const [container, target, value, formatter] of measurements) {
    $(container).hidden = !Number.isFinite(value);
    if (Number.isFinite(value)) $(target).textContent = formatter(value);
  }
  $('run-measurements').hidden = !measurements.some(([, , value]) => Number.isFinite(value));
  $('execution-scope').textContent = summary.time_scope || 'Recorded run duration';
  $('replica-count').textContent = Number.isFinite(summary.gpu_replicas) ? `${summary.gpu_replicas} shared model replicas · ${summary.model}` : '';
  const evaluation = summary.evaluation;
  $('quality-status').textContent = evaluation ? 'EVALUATED' : 'PENDING';
  const protocol = evaluation?.protocol;
  $('quality-protocol').textContent = protocol ? typeof protocol === 'string' ? protocol : Object.values(protocol).join(' ') : 'Quality evaluation pending. Worker-call completion does not imply a correct solution.';
  $('quality-values').replaceChildren();
  for (const [key, title] of [['all_tasks', 'All tasks · bulk workload'], ['humaneval', 'HumanEval'], ['mbpp_test', 'MBPP official test split'], ['mbpp_all', 'MBPP all splits · workload']]) {
    if (!evaluation?.[key]) continue;
    const result = evaluation[key];
    const card = document.createElement('div');
    const label = document.createElement('span');
    const value = document.createElement('strong');
    const detail = document.createElement('small');
    label.textContent = title;
    value.textContent = `${(result.pass_rate * 100).toFixed(2)}%`;
    detail.textContent = `${format(result.passed)} / ${format(result.tasks)} passed`;
    card.append(label, value, detail);
    $('quality-values').append(card);
  }

}

function showError(error) {
  $('error').hidden = false;
  $('error').textContent = error.message;
}

function load(data, source) {
  if (!Array.isArray(data.events) || !data.events.length) throw new Error('The replay must contain a non-empty events array.');
  for (const event of data.events) {
    if (!Number.isFinite(event.timestamp) || typeof event.type !== 'string') throw new Error('Each event must have a numeric UTC epoch timestamp and a string type.');
  }
  state.data = data;
  state.events = [...data.events].sort((a, b) => a.timestamp - b.timestamp);
  state.start = state.events[0].timestamp;
  state.end = state.events.at(-1).timestamp;
  state.cursor = state.end;
  state.playing = false;
  state.selected = null;
  state.selectedEvent = null;
  const spawns = state.events.filter((event) => event.type === 'agent_spawned');
  state.totalAgents = new Set(spawns.map((event) => event.agent_id)).size;
  state.spawnStart = state.events.find((event) => event.type === 'routing_started')?.timestamp ?? spawns[0]?.timestamp ?? state.start;
  state.spawnEnd = spawns.at(-1)?.timestamp ?? state.start;
  const metadata = {...state.events.find((event) => event.type === 'run_started')?.payload, ...data.metadata};
  renderRunMeasurements(metadata);
  $('run-description').textContent = [metadata.run_id, metadata.model || metadata.model_path, metadata.dataset, metadata.mode].filter(Boolean).join(' · ') || 'Recorded role-worker execution';
  $('source').textContent = source;
  $('error').hidden = true;
  $('empty').hidden = true;
  $('play').disabled = false;
  $('latest').disabled = false;
  $('replay-spawning').disabled = !spawns.length;
  $('spawn-checkpoint').disabled = !spawns.length;
  $('scrubber').disabled = false;
  $('duration').textContent = elapsed(state.end - state.start);
  const tasks = [...new Set(state.events.map((event) => event.task_id).filter((id) => id !== undefined && id !== null && id !== ''))];
  $('task-filter').replaceChildren(new Option('All tasks', ''), ...tasks.map((task) => new Option(String(task), String(task))));
  $('task-total').textContent = `${format(tasks.length)} tasks recorded in run`;
  render();
}

function reconstruct() {
  const agents = new Map();
  const tasks = new Set();
  const batches = [];
  const gpuIds = new Set();
  let decisions = 0;
  let peakRunning = 0;
  let running = 0;
  let count = 0;
  for (const event of state.events) {
    if (event.timestamp > state.cursor) break;
    count++;
    if (event.type === 'agent_spawned') {
      agents.set(event.agent_id, {id: event.agent_id, task: event.task_id, parent: event.parent_id, role: event.role, status: 'queued', events: [event]});
    } else if (event.agent_id && agents.has(event.agent_id)) {
      const agent = agents.get(event.agent_id);
      agent.events.push(event);
      if (event.type === 'agent_started') {
        if (agent.status !== 'running') running++;
        agent.status = 'running';
      }
      if (event.type === 'agent_completed' || event.type === 'agent_failed') {
        if (agent.status === 'running') running--;
        agent.status = event.type === 'agent_failed' || ['failed', 'error'].includes(event.payload?.status) ? 'failed' : 'completed';
      }
    }
    peakRunning = Math.max(peakRunning, running);
    if (event.type === 'decision') decisions++;
    if (event.type === 'task_completed') tasks.add(event.task_id);
    const batch = event.payload?.batch_stats || (['batch_completed', 'inference_batch'].includes(event.type) ? event.payload : null);
    if (batch) {
      batches.push(batch);
      if (batch.gpu_id !== undefined) gpuIds.add(batch.gpu_id);
      for (const id of batch.gpu_ids || []) gpuIds.add(id);
    }
  }
  state.agents = agents;
  state.count = count;
  return {tasks, batches, gpuIds, decisions, peakRunning};
}

function render() {
  if (!state.data) return;
  const result = reconstruct();
  const agents = [...state.agents.values()];
  $('spawned').textContent = format(agents.length);
  for (const status of ['running', 'queued', 'completed']) $(''+status).textContent = format(agents.filter((agent) => agent.status === status).length);
  $('decisions').textContent = format(result.decisions);
  $('tasks-completed').textContent = format(result.tasks.size);
  $('batches').textContent = result.batches.length ? format(result.batches.length) : 'Unreported';
  const sizes = result.batches.map((batch) => batch.batch_size).filter(Number.isFinite);
  $('largest-batch').textContent = sizes.length ? format(Math.max(...sizes)) : 'Unreported';
  $('gpus').textContent = result.gpuIds.size ? [...result.gpuIds].join(', ') : 'Unreported';
  $('peak-running').textContent = format(result.peakRunning);
  $('elapsed').textContent = elapsed(state.cursor - state.start);
  $('scrubber').value = state.end === state.start ? 1000 : Math.round(1000 * (state.cursor - state.start) / (state.end - state.start));
  $('play').textContent = state.playing ? 'Ⅱ Pause' : '▶ Play';
  const finished = state.events.slice(0, state.count).some((event) => event.type === 'run_completed');
  $('run-status').textContent = state.playing ? 'REPLAYING' : finished ? 'RUN COMPLETED · REPLAY' : 'RECORDED SNAPSHOT';
  $('event-position').textContent = `${format(state.count)} / ${format(state.events.length)} events`;
  $('timestamp').textContent = new Date(state.cursor * 1000).toISOString();
  draw();
  renderTimeline();
  renderInspection();
}

function draw() {
  const canvas = $('field');
  const width = $('field-scroll').clientWidth;
  const fieldHeight = $('field-scroll').clientHeight;
  state.cell = Math.max(10, Math.min(25, Math.floor(Math.sqrt((width - 2 * state.padding) * (fieldHeight - 2 * state.padding) / Math.max(1, state.totalAgents)))));
  state.columns = Math.max(1, Math.floor((width - 2 * state.padding) / state.cell));
  state.shown = [...state.agents.values()].filter((agent) => !$('task-filter').value || String(agent.task) === $('task-filter').value);
  $('visible-count').textContent = state.data ? format(state.shown.length) : '';
  const height = Math.max($('field-scroll').clientHeight - 1, Math.ceil(state.shown.length / state.columns) * state.cell + 2 * state.padding);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const size = state.cell - Math.max(3, Math.round(state.cell / 3));
  const points = new Map(state.shown.map((agent, index) => [agent.id, {x: state.padding + (index % state.columns) * state.cell, y: state.padding + Math.floor(index / state.columns) * state.cell}]));
  const selected = state.agents.get(state.selected);
  const family = new Set([state.selected, selected?.parent]);
  for (const agent of state.shown) if (agent.parent === state.selected) family.add(agent.id);
  if (selected) {
    ctx.strokeStyle = '#c7f59888';
    ctx.lineWidth = 1.5;
    for (const agent of state.shown) {
      if (agent.id !== selected.id && agent.parent !== selected.id) continue;
      const from = points.get(agent.parent);
      const to = points.get(agent.id);
      if (!from || !to) continue;
      ctx.beginPath();
      ctx.moveTo(from.x + size / 2, from.y + size / 2);
      ctx.lineTo(to.x + size / 2, to.y + size / 2);
      ctx.stroke();
    }
  }
  for (const agent of state.shown) {
    const {x, y} = points.get(agent.id);
    ctx.globalAlpha = selected && !family.has(agent.id) ? 0.35 : 1;
    ctx.fillStyle = colors[agent.status];
    ctx.beginPath();
    ctx.roundRect(x, y, size, size, agent.role === 'review' ? size / 2 : agent.role === 'repair' ? 1 : size / 4);
    ctx.fill();
    if (agent.id === state.selected) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 2, y - 2, size + 4, size + 4);
    }
  }
  ctx.globalAlpha = 1;
}

function renderTimeline() {
  const events = state.events.slice(0, state.count).filter((event) => !$('task-filter').value || String(event.task_id) === $('task-filter').value);
  $('timeline-count').textContent = `${format(events.length)} EVENTS`;
  const fragment = document.createDocumentFragment();
  for (const event of events.slice(-60).reverse()) {
    const button = document.createElement('button');
    button.className = 'timeline-row';
    const time = document.createElement('span');
    time.className = 'timeline-time';
    time.textContent = elapsed(event.timestamp - state.start);
    const type = document.createElement('span');
    type.className = 'timeline-type';
    type.textContent = event.type.replaceAll('_', ' ');
    const name = document.createElement('span');
    name.className = 'timeline-name';
    name.textContent = [event.task_id, event.agent_id, event.role].filter((value) => value !== undefined && value !== null).join(' / ') || 'Run event';
    button.append(time, type, name);
    button.addEventListener('click', () => {
      state.selected = event.agent_id || null;
      state.selectedEvent = event;
      state.tab = event.type === 'decision' ? 'decisions' : 'events';
      draw();
      renderInspection();
    });
    fragment.append(button);
  }
  $('timeline').replaceChildren(fragment);
}

function renderInspection() {
  const heading = $('selection-heading');
  const inspection = $('inspection');
  heading.replaceChildren();
  inspection.replaceChildren();
  document.querySelectorAll('[data-tab]').forEach((button) => button.classList.toggle('active', button.dataset.tab === state.tab));
  const agent = state.agents.get(state.selected);
  const event = state.selectedEvent?.timestamp <= state.cursor ? state.selectedEvent : null;
  if (!agent && !event) {
    heading.textContent = 'Select a worker call or timeline event.';
    return;
  }
  const title = document.createElement('strong');
  title.textContent = agent?.id || event.type;
  const subtitle = document.createElement('small');
  subtitle.textContent = agent ? `${agent.role} · ${agent.status} · ${agent.task}${agent.parent ? ` · parent: ${agent.parent}` : ''}` : `Event ${event.event_id} · ${event.task_id || 'run'}`;
  heading.append(title, subtitle);
  let events;
  if (state.tab === 'decisions') {
    const task = agent?.task ?? event.task_id;
    events = state.events.slice(0, state.count).filter((item) => item.type === 'decision' && (item === event || (task !== undefined && item.task_id === task)));
  } else if (state.tab === 'output') {
    events = agent ? agent.events.filter((item) => ['agent_completed', 'agent_failed'].includes(item.type)) : [event];
  } else {
    events = agent ? agent.events : [event];
  }
  if (!events.length) {
    const note = document.createElement('p');
    note.className = 'subtle';
    note.textContent = 'No matching evidence at this replay position.';
    inspection.append(note);
  }
  for (const item of events) {
    const label = document.createElement('div');
    label.className = 'evidence-label';
    label.textContent = `${elapsed(item.timestamp - state.start)} · ${item.type} · ${item.event_id}`;
    const pre = document.createElement('pre');
    pre.textContent = pretty(state.tab === 'events' ? item : state.tab === 'output' && typeof item.payload?.output === 'string' ? item.payload.output : item.payload || {});
    inspection.append(label, pre);
  }
}

function tick(now) {
  if (state.playing) {
    state.cursor = Math.min(state.end, state.cursor + (now - state.frame) / 1000 * Number($('speed').value));
    if (state.cursor >= state.end) state.playing = false;
    if (!state.playing || now - state.lastRender >= 100) {
      state.lastRender = now;
      render();
    }
  }
  state.frame = now;
  requestAnimationFrame(tick);
}

$('play').addEventListener('click', () => {
  if (!state.playing && state.cursor >= state.end) state.cursor = state.start;
  state.playing = !state.playing;
  state.frame = performance.now();
  render();
});
$('latest').addEventListener('click', () => {state.playing = false; state.cursor = state.end; render();});
$('replay-spawning').addEventListener('click', () => {state.cursor = state.spawnStart; state.playing = true; state.selected = null; state.selectedEvent = null; $('task-filter').value = ''; state.frame = performance.now(); render();});
$('spawn-checkpoint').addEventListener('click', () => {state.playing = false; state.cursor = state.spawnEnd; state.selected = null; state.selectedEvent = null; $('task-filter').value = ''; render();});
$('scrubber').addEventListener('input', () => {state.playing = false; state.cursor = state.start + Number($('scrubber').value) / 1000 * (state.end - state.start); render();});
$('task-filter').addEventListener('change', () => {state.selected = null; state.selectedEvent = null; render();});
document.querySelectorAll('[data-tab]').forEach((button) => button.addEventListener('click', () => {state.tab = button.dataset.tab; renderInspection();}));
$('field').addEventListener('click', (event) => {
  const rect = event.currentTarget.getBoundingClientRect();
  const column = Math.floor((event.clientX - rect.left - state.padding) / state.cell);
  const row = Math.floor((event.clientY - rect.top - state.padding) / state.cell);
  if (column < 0 || column >= state.columns || row < 0) return;
  const agent = state.shown[row * state.columns + column];
  if (!agent) return;
  state.selected = agent.id;
  state.selectedEvent = null;
  state.tab = 'output';
  draw();
  renderInspection();
});
$('file').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {load(JSON.parse(await file.text()), file.name);} catch (error) {showError(error);}
});
new ResizeObserver(draw).observe($('field-scroll'));
requestAnimationFrame(tick);
const embeddedReplay = document.getElementById('embedded-replay');
if (embeddedReplay) {
  load(JSON.parse(embeddedReplay.textContent), 'Embedded recorded run');
} else fetch('./replay.json').then(async (response) => {
  if (response.status === 404) return;
  if (!response.ok) throw new Error(`Could not load replay.json: HTTP ${response.status}`);
  load(await response.json(), 'replay.json');
}).catch((error) => {
  $('run-description').textContent = location.protocol === 'file:' ? 'Use Open run to load your replay JSON, or serve this directory over HTTP.' : 'Could not load replay.json. Use Open run to select a recorded execution.';
  if (location.protocol !== 'file:') showError(error);
});
