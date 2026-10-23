#!/usr/bin/env node
// @ts-check
import { execFileSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { defaultIdentityDir, loadOrCreateIdentity, publicKeyFromSeed } from '../src/identity.js'
import { addMember, createTeam, parseTeam, readTeam, readTeamFile, watchTeamFile, writeTeamFile } from '../src/team.js'
import { createLogger } from '../src/log.js'
import { DEFAULT_CONFIG, labelsFrom, validateConfig } from '../src/config.js'
import { openNode } from '../src/node.js'
import { createRun, mergedPath, runRounds } from '../src/run.js'
import { fakeTrainer, pythonTrainer } from '../src/trainer.js'

const USAGE = `potlucktune: fine-tune one LoRA adapter on your team's data without moving the data

Usage:
  potlucktune try                              Try it on one machine: three members, a fake trainer,
                                               connected over the real DHT. No Python needed.
  potlucktune id                               Print your public key (creates it on first run)
  potlucktune team init <team-name> [--member <name>]
                                               Create a team file with you as the first member
  potlucktune team add <name> <public-key>     Add a member to the team file
  potlucktune init --labels-from <file.jsonl> [--config <file.json>]
                                               Owner: start a training run for the team file's members
  potlucktune train --data <file.jsonl> [--rounds <n>]
                                               Train on your data and merge with the team, round by round
  potlucktune status                           Rounds, posts and merged hashes from the last 'train'
  potlucktune eval --data <file.jsonl> [--round <n>]
                                               Accuracy of a merged adapter (default: the latest)
  potlucktune export --out <dir> [--round <n>] Write the merged adapter as a standard PEFT adapter

Options:
  --team <file>          Team file (default: ./team.json). Same format as PotluckLLM's.
  --member <name>        Your name in a new team file (default: this machine's hostname)
  --data <file>          JSONL, one {"text": ..., "label": ...} per line. Stays on this machine.
  --rounds <n>           Rounds to train (default: 3)
  --round <n>            'eval', 'export': which round's merge to use (default: latest)
  --labels-from <file>   'init': take the label list from this JSONL file
  --config <file>        'init': JSON that overrides the default training config
  --python <path>        Python with torch, transformers, peft (default: $POTLUCKTUNE_PYTHON or python3)
  --device <d>           auto, cuda or cpu (default: auto)
  --linger <s>           'train': after the last round, wait this long for the others (default: 120)
  --baseline            'eval': also evaluate the base model with no adapter
  --bootstrap <list>     DHT bootstrap nodes as host:port,host:port (for local testing)
  -h, --help             Show this help
  -v, --version          Show the version

Your identity is PotluckLLM's (~/.potluck), so one team file serves all three tools.
Run data (adapters, merges, the shared log) lives in ~/.potlucktune.`

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    team: { type: 'string', default: 'team.json' },
    member: { type: 'string' },
    data: { type: 'string' },
    rounds: { type: 'string', default: '3' },
    round: { type: 'string' },
    'labels-from': { type: 'string' },
    config: { type: 'string' },
    python: { type: 'string' },
    device: { type: 'string', default: 'auto' },
    linger: { type: 'string', default: '120' },
    out: { type: 'string' },
    'baseline': { type: 'boolean', default: false },
    bootstrap: { type: 'string' },
    help: { type: 'boolean', short: 'h' },
    version: { type: 'boolean', short: 'v' }
  }
})

const log = createLogger()

async function main () {
  const [command, ...args] = positionals

  if (values.version) {
    const pkg = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'))
    console.log(pkg.version)
    return
  }
  if (values.help || !command) {
    console.log(USAGE)
    return
  }

  if (command === 'id') {
    const identity = loadOrCreateIdentity()
    if (identity.created) console.error(`Created a new identity in ${defaultIdentityDir()}`)
    console.log(identity.publicKey.toString('hex'))
    return
  }
  if (command === 'team') return teamCommand(args)
  if (command === 'init') return init()
  if (command === 'train') return train()
  if (command === 'status') return status()
  if (command === 'eval') return evaluate()
  if (command === 'export') return exportAdapter()
  if (command === 'try') return tryLocally()

  throw new Error(`unknown command "${command}". Run: potlucktune --help`)
}

/**
 * Where a team's run data lives: one folder per team topic.
 *
 * @param {Buffer} topic
 */
function runDir (topic) {
  const root = path.join(os.homedir(), '.potlucktune')
  fs.mkdirSync(root, { recursive: true, mode: 0o700 })
  // Adapters can reveal facts about the data they were trained on.
  const mode = fs.statSync(root).mode & 0o777
  if (mode & 0o077) throw new Error(`${root} is accessible by other users (mode ${mode.toString(8)}). Run: chmod 700 ${root}`)
  const dir = path.join(root, topic.toString('hex').slice(0, 16))
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
  return dir
}

/** @param {string} file */
function baseKeyFrom (file) {
  const raw = readTeamFile(file)
  const hex = /** @type {any} */ (raw).potlucktune?.base
  if (typeof hex !== 'string' || !/^[0-9a-f]{64}$/.test(hex)) return null
  return Buffer.from(hex, 'hex')
}

function trainerFor (/** @type {string} */ dir) {
  if (!['auto', 'cuda', 'cpu'].includes(values.device ?? '')) throw new Error('--device must be auto, cuda or cpu')
  return pythonTrainer({ python: values.python, device: values.device, workDir: path.join(dir, 'tmp') })
}

async function init () {
  const { seed, publicKey } = loadOrCreateIdentity()
  const team = readTeam(values.team)
  if (!team.isMember(publicKey)) throw new Error(`your key is not in ${values.team}`)
  if (baseKeyFrom(values.team)) throw new Error(`${values.team} already has a training run. Use a new team file for a new run.`)

  let config = { ...DEFAULT_CONFIG }
  if (values.config) config = { ...config, ...JSON.parse(fs.readFileSync(values.config, 'utf8')) }
  if (values['labels-from']) config.labels = labelsFrom(fs.readFileSync(values['labels-from'], 'utf8'))
  validateConfig(config)

  const dir = runDir(team.topic)
  const node = await openNode({ storage: path.join(dir, 'store'), seed, team, baseKey: null, bootstrap: parseBootstrap(values.bootstrap), log })
  try {
    console.error(`Computing the frozen A for ${config.model} (downloads the model on first use)...`)
    const aHash = await createRun({ node, trainer: trainerFor(dir), name: team.name, members: team.members, config })
    const raw = readTeamFile(values.team)
    writeTeamFile(values.team, { ...raw, potlucktune: { base: node.base.key.toString('hex') } })
    console.log(`Started a training run for "${team.name}" with ${team.members.length} members and ${config.labels.length} labels.`)
    console.log(`Model ${config.model}, rank ${config.rank}, ${config.steps} steps per round, frozen A ${aHash.slice(0, 12)}.`)
    console.log(`Send the updated ${values.team} to every member, then each runs: potlucktune train --data <their file>`)
  } finally {
    await node.close()
  }
}

async function train () {
  if (!values.data) throw new Error('usage: potlucktune train --data <file.jsonl> [--rounds <n>]')
  if (!fs.existsSync(values.data)) throw new Error(`${values.data} not found`)
  const rounds = positiveInt(values.rounds, '--rounds')
  const { seed, publicKey } = loadOrCreateIdentity()
  const team = readTeam(values.team)
  if (!team.isMember(publicKey)) throw new Error(`your key is not in ${values.team}`)
  const baseKey = baseKeyFrom(values.team)
  if (!baseKey) throw new Error(`${values.team} has no training run. The owner starts one with: potlucktune init`)

  const dir = runDir(team.topic)
  const statusFile = path.join(dir, 'status.json')
  /** @type {{ events: { time: string, event: string, fields: Record<string, unknown> }[] }} */
  const status = { events: [] }
  const trackingLog = (/** @type {string} */ event, /** @type {Record<string, unknown>} */ fields = {}) => {
    log(event, fields)
    status.events.push({ time: new Date().toISOString(), event, fields })
    if (status.events.length > 500) status.events.shift()
    fs.writeFileSync(statusFile, JSON.stringify(status, null, 2), { mode: 0o600 })
  }

  const node = await openNode({ storage: path.join(dir, 'store'), seed, team, baseKey, bootstrap: parseBootstrap(values.bootstrap), log: trackingLog })
  const stopWatch = watchTeamFile(values.team, (t) => {
    node.setTeam(t)
    log('team-reloaded', { members: t.members.length })
  }, (err) => log('team-file-error', { error: err.message }))

  const controller = new AbortController()
  process.once('SIGINT', () => controller.abort())
  process.once('SIGTERM', () => controller.abort())
  try {
    await syncMembers(node, team, publicKey)
    const report = await runRounds({
      node,
      trainer: trainerFor(dir),
      data: path.resolve(values.data),
      rounds,
      dir,
      log: trackingLog,
      signal: controller.signal,
      lingerMs: positiveInt(values.linger, '--linger') * 1000
    })
    const last = report.at(-1)
    if (last) console.log(`Round ${last.round} merged: ${last.sha256.slice(0, 12)} (${last.reported} of ${node.state.members.size} members reported, ${report.every((r) => r.agreed) ? 'all agree' : 'HASHES DIFFER'}).`)
    console.log(`Merged adapters are in ${path.join(dir, 'rounds')}. Next: potlucktune eval --data <test file>`)
  } catch (err) {
    if (controller.signal.aborted) console.error('Stopped. Run the same command again to continue where it left off.')
    else throw err
  } finally {
    stopWatch()
    await node.close()
  }
}

/**
 * The owner keeps the run's member list in step with the team file.
 *
 * @param {import('../src/node.js').TuneNode} node
 * @param {import('../src/team.js').Team} team
 * @param {Buffer} publicKey
 */
async function syncMembers (node, team, publicKey) {
  await node.base.update()
  const state = await node.refresh()
  if (!state.team || state.team.owner !== publicKey.toString('hex')) return
  const current = [...state.members.keys()].sort().join()
  const wanted = team.members.map((m) => m.key).sort().join()
  if (current === wanted) return
  await node.append({ type: 'members', members: team.members })
  log('members-updated', { members: team.members.length })
}

function status () {
  const team = readTeam(values.team)
  const file = path.join(runDir(team.topic), 'status.json')
  if (!fs.existsSync(file)) {
    console.log('No training on this machine yet. Run: potlucktune train --data <file>')
    return
  }
  const { events } = JSON.parse(fs.readFileSync(file, 'utf8'))
  for (const e of events.filter((/** @type {any} */ e) => ['ready', 'posted', 'merged', 'agree', 'DISAGREE', 'waiting'].includes(e.event)).slice(-30)) {
    const fields = Object.entries(e.fields).map(([k, v]) => `${k}=${v}`).join(' ')
    console.log(`${e.time}  ${e.event.padEnd(8)} ${fields}`)
  }
}

/** @param {string} dir */
function latestRound (dir) {
  const rounds = fs.existsSync(path.join(dir, 'rounds'))
    ? fs.readdirSync(path.join(dir, 'rounds')).map(Number).filter((r) => Number.isInteger(r) && fs.existsSync(mergedPath(dir, r)))
    : []
  return rounds.length ? Math.max(...rounds) : null
}

/** @param {string} dir */
function storedConfig (dir) {
  const file = path.join(dir, 'config.json')
  if (!fs.existsSync(file)) throw new Error('no training config on this machine yet. Run potlucktune train first.')
  return validateConfig(JSON.parse(fs.readFileSync(file, 'utf8')))
}

async function evaluate () {
  if (!values.data) throw new Error('usage: potlucktune eval --data <file.jsonl> [--round <n>]')
  const team = readTeam(values.team)
  const dir = runDir(team.topic)
  const config = storedConfig(dir)
  const round = values.round ? positiveInt(values.round, '--round') : latestRound(dir)
  if (!round || !fs.existsSync(mergedPath(dir, round))) throw new Error('no merged adapter for that round on this machine')
  const trainer = trainerFor(dir)
  if (values['baseline']) {
    const base = await trainer.evaluate({ config, data: values.data, adapter: null })
    console.log(`base model, no adapter: accuracy ${base.accuracy} on ${base.examples} examples`)
  }
  const own = path.join(dir, 'rounds', String(round), 'own.safetensors')
  if (fs.existsSync(own)) {
    const mine = await trainer.evaluate({ config, data: values.data, adapter: own })
    console.log(`round ${round}, your adapter before merging: accuracy ${mine.accuracy}`)
  }
  const res = await trainer.evaluate({ config, data: values.data, adapter: mergedPath(dir, round) })
  console.log(`round ${round}, merged adapter: accuracy ${res.accuracy} on ${res.examples} examples`)
}

async function exportAdapter () {
  if (!values.out) throw new Error('usage: potlucktune export --out <dir> [--round <n>]')
  const team = readTeam(values.team)
  const dir = runDir(team.topic)
  const round = values.round ? positiveInt(values.round, '--round') : latestRound(dir)
  if (!round) throw new Error('no merged adapter on this machine yet')
  await trainerFor(dir).exportAdapter({ config: storedConfig(dir), adapter: mergedPath(dir, round), out: values.out })
  console.log(`Wrote round ${round}'s merged adapter to ${values.out}. Load it with peft.PeftModel.from_pretrained(base_model, "${values.out}").`)
}

async function tryLocally () {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'potlucktune-try-'))
  const bootstrap = parseBootstrap(values.bootstrap)
  const names = ['alice', 'bob', 'carol']
  const people = names.map((name) => {
    const seed = crypto.randomBytes(32)
    return { name, seed, key: publicKeyFromSeed(seed).toString('hex') }
  })
  const team = parseTeam({ name: 'try', topic: crypto.randomBytes(32).toString('hex'), members: people.map(({ name, key }) => ({ name, key })) })
  const trainer = fakeTrainer({ delayMs: 300 })
  const config = { ...DEFAULT_CONFIG, labels: ['billing', 'login', 'shipping'] }
  const rounds = 3

  console.log('Three members on this machine, each with their own data, connecting over the public DHT.')
  console.log('The trainer is a stand-in (no model, no Python); the networking, merging and checks are real.\n')

  /** @param {string} who */
  const memberLog = (who) => (/** @type {string} */ event, /** @type {Record<string, unknown>} */ fields = {}) => {
    if (!['ready', 'posted', 'merged', 'agree', 'DISAGREE', 'connect'].includes(event)) return
    const text = Object.entries(fields).filter(([k]) => ['round', 'peer', 'members', 'sha', 'examples', 'reported'].includes(k)).map(([k, v]) => `${k}=${v}`).join(' ')
    console.log(`${who.padEnd(6)} ${event.padEnd(7)} ${text}`)
  }

  const nodes = []
  try {
    const owner = await openNode({ storage: path.join(tmp, 'alice'), seed: people[0].seed, team, baseKey: null, bootstrap, log: memberLog('alice') })
    nodes.push(owner)
    await createRun({ node: owner, trainer, name: 'try', members: team.members, config })
    for (const p of people.slice(1)) {
      nodes.push(await openNode({ storage: path.join(tmp, p.name), seed: p.seed, team, baseKey: owner.base.key, bootstrap, log: memberLog(p.name) }))
    }
    const data = people.map((p, i) => {
      const file = path.join(tmp, `${p.name}.jsonl`)
      const rows = Array.from({ length: 20 + i * 10 }, (_, j) => JSON.stringify({ text: `${p.name} example ${j}`, label: config.labels[i] }))
      fs.writeFileSync(file, rows.join('\n') + '\n')
      return file
    })
    const reports = await Promise.all(nodes.map((node, i) => runRounds({
      node, trainer, data: data[i], rounds, dir: path.join(tmp, `${people[i].name}-run`), log: memberLog(people[i].name), lingerMs: 30000
    })))
    const hashes = new Set(reports.map((r) => r.at(-1)?.sha256))
    console.log(`\n${hashes.size === 1 ? 'All three members' : 'Members did NOT'} end with the same merged adapter${hashes.size === 1 ? `: ${[...hashes][0]?.slice(0, 12)}` : ''}.`)
    console.log('No member sent its data to anyone; only adapters (a few KB here) crossed the network.')
  } finally {
    // One at a time: closing every swarm at once has hung on the public DHT.
    for (const n of nodes) await n.close()
    fs.rmSync(tmp, { recursive: true, force: true })
  }
}

/** @param {string[]} args */
function teamCommand ([sub, ...args]) {
  if (sub === 'init') {
    const [teamName] = args
    if (!teamName) throw new Error('usage: potlucktune team init <team-name> [--member <name>]')
    const { publicKey } = loadOrCreateIdentity()
    const member = values.member ?? os.hostname()
    try {
      writeTeamFile(values.team, createTeam(teamName, { name: member, key: publicKey.toString('hex') }), { create: true })
    } catch (err) {
      if (err.code === 'EEXIST') throw new Error(`${values.team} already exists. Delete it first or pass --team <other file>.`)
      throw err
    }
    console.log(`Created ${values.team} for team "${teamName}" with you as "${member}".`)
    if (trackedByGit(values.team)) {
      console.error(`Warning: ${values.team} is inside a git repository and not ignored. Add it to .gitignore so the topic and member list are never committed.`)
    }
    console.log('Add members with: potlucktune team add <name> <public-key>')
    return
  }

  if (sub === 'add') {
    const [name, key] = args
    if (!name || !key) throw new Error('usage: potlucktune team add <name> <public-key>')
    writeTeamFile(values.team, addMember(readTeamFile(values.team), name, key.toLowerCase()))
    console.log(`Added "${name}" (${key.slice(0, 8)}) to ${values.team}. Share the updated file with the team.`)
    return
  }

  throw new Error('usage: potlucktune team init <team-name> | potlucktune team add <name> <public-key>')
}

/**
 * True when `file` is inside a git work tree and git would not ignore it.
 *
 * @param {string} file
 */
function trackedByGit (file) {
  try {
    execFileSync('git', ['check-ignore', '-q', file], { stdio: 'ignore' })
    return false
  } catch (err) {
    // Exit code 1: not ignored. 128: not a git repository.
    return err.status === 1
  }
}

/** @param {string | undefined} list */
function parseBootstrap (list) {
  if (!list) return undefined
  return list.split(',').map((entry) => {
    const [host, port] = entry.split(':')
    return { host, port: Number(port) }
  })
}

/**
 * @param {string} value
 * @param {string} flag
 */
function positiveInt (value, flag) {
  const n = Number(value)
  if (!Number.isInteger(n) || n < 1) throw new Error(`${flag} must be a positive integer`)
  return n
}

main().catch((err) => {
  if (err.code === 'ENOENT' && err.path) err.message = `${err.path} not found${err.path.endsWith('.json') ? '. Create a team with: potlucktune team init <name>' : ''}`
  console.error(`potlucktune: ${err.message}`)
  process.exit(1)
})
