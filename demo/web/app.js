const $ = (id) => document.getElementById(id);
const state = {sample: null, samples: [], rounds: [], nodes: new Map(), source: null, actions: 0, status: 'loading', config: {}};

function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatted(value) {
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function observationText(value) {
  return typeof value === 'object' && value !== null && 'observation' in value
    ? formatted(value.observation) : formatted(value);
}

function initialText(context) {
  return context.split('\nExecution interface (mandatory argument keys, JSON types, and tool signatures):')[0].trim();
}

function setStatus(status, label) {
  state.status = status;
  $('run-status').textContent = label;
  $('run-status').classList.toggle('running', status === 'running');
  document.body.dataset.status = status;
}

function stopStream() {
  if (state.source) state.source.close();
  state.source = null;
  $('run').disabled = false;
  $('stop').hidden = true;
  $('mode').disabled = false;
}

function showError(message) {
  stopStream();
  setStatus('error', 'Run failed');
  $('error').textContent = message;
  $('error').hidden = false;
}

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function showState(branch, value, caption) {
  $('observed-branch').textContent = branch;
  $('observation').textContent = observationText(value);
  $('observation').scrollTop = 0;
  $('state-caption').textContent = caption;
}

function resetView(sample) {
  state.rounds = [];
  state.nodes.clear();
  state.actions = 0;
  $('round-count').textContent = '—';
  $('action-count').textContent = '0';
  $('frontier-count').textContent = '—';
  $('result').hidden = true;
  $('error').hidden = true;
  $('timeline').replaceChildren();
  const empty = element('div', 'empty-state');
  empty.append(element('div', 'empty-glyph', '↗'), element('h3', '', 'Watch inference unfold'), element('p', '', 'Each round shows branch selection, structured actions, and actual observations.'));
  $('timeline').append(empty);
  $('declaration').replaceChildren(element('p', 'muted', 'The model declares finite action fields, then the runtime scores and executes structured choices.'));
  $('declaration-status').textContent = 'Awaiting round';
  $('full-context').textContent = sample.context;
  showState('Initial task', initialText(sample.context), 'Initial task text · environment instructions');
}

async function selectSample(id) {
  stopStream();
  $('run').disabled = true;
  setStatus('loading', 'Loading task');
  const sample = await getJSON(`/api/samples/${encodeURIComponent(id)}`);
  state.sample = sample;
  document.body.dataset.sample = sample.id;
  $('title').textContent = sample.title;
  $('dataset').textContent = sample.dataset;
  $('task-id').textContent = `Task ${sample.task_id}`;
  document.querySelectorAll('.sample').forEach(button => {
    button.classList.toggle('active', button.dataset.id === sample.id);
    button.setAttribute('aria-current', button.dataset.id === sample.id ? 'true' : 'false');
  });
  resetView(sample);
  setStatus('ready', 'Ready to explore');
  $('run').disabled = false;
}

function renderDeclaration(declaration) {
  const target = $('declaration');
  target.replaceChildren();
  for (const field of declaration.fields) {
    const row = element('div', 'field');
    const values = element('div', '');
    values.append(element('span', 'field-label', field.question), element('span', 'field-values', field.values.map(value => JSON.stringify(value)).join('  ·  ')));
    row.append(element('span', 'field-name', field.id), values);
    target.append(row);
  }
  target.append(element('pre', 'template', `${declaration.action.tool}(${JSON.stringify(declaration.action.arguments)})`));
  $('declaration-status').textContent = `${declaration.fields.length} finite fields`;
}

function renderRound(round) {
  const timeline = $('timeline');
  if (!state.rounds.length) timeline.replaceChildren();
  const previous = state.rounds.at(-1);
  state.rounds.push(round);
  const selected = round.selected[0];
  const selectedNode = state.nodes.get(selected);
  if (selectedNode) showState(selected, selectedNode.observation, 'Selected branch · exact environment return');
  if (selected === 'root') showState('root', initialText(state.sample.context), 'Selected branch · initial task text');
  document.querySelectorAll('.branch').forEach(node => {
    if (node.dataset.branch === selected) {
      node.classList.add('selected');
      node.querySelector('.branch-state').textContent = 'Selected in round ' + (round.turn + 1);
    }
  });
  if (round.revision && round.revision.declaration) renderDeclaration(round.revision.declaration);
  const card = element('article', 'turn');
  card.dataset.turn = round.turn;
  const heading = element('div', 'turn-heading');
  heading.append(element('span', 'turn-title', `Round ${String(round.turn + 1).padStart(2, '0')}`));
  if (round.selected_operation) heading.append(element('span', 'operation', round.selected_operation));
  const recovery = previous && selected !== previous.selected[0] && !Object.hasOwn(previous.children || {}, selected);
  if (recovery) heading.append(element('span', 'operation recovery', 'Return to retained branch'));
  const probability = round.frontier_decision.probabilities[round.frontier_decision.option_ids.indexOf(selected)];
  const selectionLabel = round.frontier_decision.option_ids.length > 1 ? `selected ${selected} · p = ${probability.toFixed(3)}` : `initial state ${selected}`;
  heading.append(element('span', 'selected-detail', selectionLabel));
  card.append(heading);
  const grid = element('div', 'spawn-grid');
  for (const [id, steps] of Object.entries(round.children || {})) {
    const metadata = steps.at(-1);
    const actionSteps = steps.filter(step => Array.isArray(step.actions));
    const actions = actionSteps.flatMap(step => step.actions);
    const observations = actionSteps.flatMap(step => step.observations);
    const observation = observations.at(-1).value;
    state.nodes.set(id, {parent: metadata.parent_id, observation});
    state.actions += actions.length;
    const branch = element('div', 'branch');
    branch.dataset.branch = id;
    branch.tabIndex = 0;
    branch.setAttribute('role', 'button');
    branch.setAttribute('aria-label', `Inspect branch ${id} and its exact observation`);
    const top = element('div', 'branch-top');
    top.append(element('span', '', `${metadata.parent_id} → ${id}`), element('span', '', `π ${Math.exp(metadata.block_conditional_log_probability).toFixed(3)}`));
    const actionText = actions.map(action => Object.entries(action.arguments).map(([key, value]) => `${key}: ${formatted(value)}`).join(', ')).join(' → ');
    branch.append(top, element('div', 'branch-action', actionText), element('div', 'branch-state', observation.done ? 'Environment stopped' : 'Observation returned ↗'));
    const inspect = () => {
      const existing = grid.querySelector('.branch-inspector');
      if (existing) existing.remove();
      const detail = element('pre', 'branch-inspector', JSON.stringify({branch: id, actions, observations: observations.map(item => item.value), selected_values: metadata.selected_values}, null, 2));
      grid.append(detail);
      showState(id, observation, 'Inspected branch · exact environment return');
    };
    branch.addEventListener('click', inspect);
    branch.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        inspect();
      }
    });
    grid.append(branch);
  }
  card.append(grid);
  if (!grid.childElementCount) card.append(element('p', 'turn-message', round.selected_operation === 'submit' ? `Submit the observed trajectory from ${selected}.` : 'No environment action executed in this round.'));
  const audit = element('details', 'turn-audit');
  audit.append(element('summary', '', 'View complete round record'));
  audit.addEventListener('toggle', () => {
    if (audit.open && !audit.querySelector('pre')) audit.append(element('pre', '', JSON.stringify(round, null, 2)));
  });
  card.append(audit);
  timeline.append(card);
  $('round-count').textContent = String(round.turn + 1);
  $('action-count').textContent = String(state.actions);
  $('frontier-count').textContent = String(round.active_frontier.length);
  if ($('follow').checked) requestAnimationFrame(() => { timeline.scrollTop = timeline.scrollHeight; });
}

function renderResult(result) {
  stopStream();
  setStatus('complete', 'Trace complete');
  const target = $('result');
  target.replaceChildren();
  target.hidden = false;
  const main = element('div', 'result-main');
  const scored = result.success === null || result.success === undefined;
  const title = scored ? 'Trajectory submitted · score-based task' : result.success ? 'Task solved · verified recorded outcome' : 'Trajectory submitted · task not solved';
  main.append(element('div', 'result-title', title));
  main.append(element('div', 'result-answer', `Final answer · ${JSON.stringify(result.answer)}`));
  const score = element('div', 'result-score', String(result.score));
  score.append(element('small', '', state.sample.score_label));
  target.append(element('div', 'result-icon', result.success === false ? '•' : '✓'), main, score);
  const selected = state.nodes.get(result.selected_terminal);
  if (selected) showState(result.selected_terminal, selected.observation, 'Submitted branch · exact environment return');
  const path = new Set();
  let node = result.selected_terminal;
  while (state.nodes.has(node)) {
    path.add(node);
    node = state.nodes.get(node).parent;
  }
  document.querySelectorAll('.branch').forEach(branch => {
    branch.classList.toggle('selected', path.has(branch.dataset.branch));
    if (path.has(branch.dataset.branch)) branch.querySelector('.branch-state').textContent = 'On submitted trajectory';
  });
  if ($('follow').checked) requestAnimationFrame(() => { $('timeline').scrollTop = $('timeline').scrollHeight; });
}

function run(mode = $('mode').value) {
  stopStream();
  resetView(state.sample);
  $('mode').value = mode;
  $('mode').disabled = true;
  $('run').disabled = true;
  $('stop').hidden = false;
  setStatus('running', mode === 'replay' ? 'Playing recorded trace' : 'Inference running');
  $('mode-note').textContent = mode === 'replay' ? `Recorded playback · ${state.config.replay_interval_ms / 1000}s per round · no model calls` : 'Live inference · executing model and environment';
  const source = new EventSource(`/api/stream/${encodeURIComponent(state.sample.id)}?mode=${encodeURIComponent(mode)}`);
  state.source = source;
  source.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === 'start') {
      state.sample = {...state.sample, ...message.sample};
      $('full-context').textContent = state.sample.context;
      showState('root', initialText(state.sample.context), 'Initial task text · environment instructions');
    }
    if (message.type === 'round') renderRound(message.round);
    if (message.type === 'result') {
      renderResult(message);
      if (mode === 'live') {
        setStatus('complete', 'Inference complete');
        const title = $('result').querySelector('.result-title');
        title.textContent = title.textContent.replace('verified recorded outcome', 'verified live outcome');
      }
    }
    if (message.type === 'error') showError(message.message);
  };
  source.onerror = () => showError('The event stream disconnected before completion. Check the server log, then start a new run.');
}

async function initialize() {
  const [samples, config] = await Promise.all([getJSON('/api/samples'), getJSON('/api/config')]);
  state.samples = samples;
  state.config = config;
  $('sample-count').textContent = String(samples.length).padStart(2, '0');
  $('mode').querySelector('[value="live"]').disabled = !config.live_enabled;
  if (config.model_label) $('provenance').textContent = config.model_label;
  samples.forEach((sample, index) => {
    const button = element('button', 'sample');
    button.dataset.id = sample.id;
    const label = element('span', '');
    label.append(element('span', 'sample-title', sample.title), element('span', 'sample-meta', `${sample.turns} rounds · ${sample.score_label} ${sample.score}`));
    button.append(element('span', 'sample-index', String(index + 1).padStart(2, '0')), label, element('span', 'sample-check', sample.success === true ? '✓' : '↗'));
    button.addEventListener('click', () => selectSample(sample.id).catch(error => showError(error.message)));
    $('samples').append(button);
  });
  $('mode-note').textContent = `Recorded playback · ${config.replay_interval_ms / 1000}s per round · no model calls`;
  await selectSample(samples[0].id);
}

$('run').addEventListener('click', () => run());
$('stop').addEventListener('click', () => {
  stopStream();
  setStatus('stopped', 'Stopped by user');
});
$('mode').addEventListener('change', () => {
  const replay = $('mode').value === 'replay';
  $('run').textContent = replay ? '▶ Play trace' : '▶ Run inference';
  $('mode-note').textContent = replay ? `Recorded playback · ${state.config.replay_interval_ms / 1000}s per round · no model calls` : 'Live inference · executing model and environment';
});
$('context-button').addEventListener('click', () => $('context-dialog').showModal());
$('close-context').addEventListener('click', () => $('context-dialog').close());
window.demo = {selectSample, run, get status() { return state.status; }, get sample() { return state.sample; }, get rounds() { return state.rounds.length; }};
initialize().catch(error => showError(error.message));
