// @ts-check
import fs from 'node:fs'
import path from 'node:path'
import { validateConfig } from './config.js'
import { mergeAdapters, sha256 } from './merge.js'
import { roundAgreement, roundPosts } from './state.js'

/**
 * @typedef {import('./node.js').TuneNode} TuneNode
 * @typedef {import('./trainer.js').Trainer} Trainer
 * @typedef {import('./state.js').State} State
 * @typedef {(event: string, fields?: Record<string, unknown>) => void} Log
 */

/**
 * Resolves with the first truthy result of `check(state)`, re-checking on
 * every view update and at least once a second.
 *
 * @template T
 * @param {TuneNode} node
 * @param {(state: State) => T | null | undefined | false} check
 * @param {{ signal?: AbortSignal, timeoutMs?: number, onWait?: (state: State) => void }} [opts]
 * @returns {Promise<T>}
 */
export async function waitFor (node, check, { signal, timeoutMs = Infinity, onWait } = {}) {
  const deadline = Date.now() + timeoutMs
  let lastNote = 0
  while (true) {
    signal?.throwIfAborted()
    await node.base.update()
    const state = await node.refresh()
    const value = check(state)
    if (value) return value
    if (Date.now() > deadline) throw new Error('timed out')
    if (onWait && Date.now() - lastNote > 10000) {
      lastNote = Date.now()
      onWait(state)
    }
    await new Promise((resolve) => {
      const timer = setTimeout(done, 1000)
      node.base.once('update', done)
      function done () {
        clearTimeout(timer)
        node.base.off('update', done)
        resolve(undefined)
      }
    })
  }
}

/**
 * Creates the training run on the owner's base: the member list, the shared
 * training config, and the hash of the frozen A it implies.
 *
 * @param {{ node: TuneNode, trainer: Trainer, name: string, members: { name: string, key: string }[], config: Record<string, unknown> }} opts
 */
export async function createRun ({ node, trainer, name, members, config }) {
  await node.refresh()
  if (node.state.team) throw new Error('this base already has a team')
  validateConfig(config)
  const aHash = await trainer.aHash(config)
  await node.append({ type: 'team', name, owner: node.identity.publicKey.toString('hex'), members, config, aHash })
  await node.refresh()
  if (!node.state.team) throw new Error('the team entry was rejected. Is your key in the team file?')
  return aHash
}

/** @param {string} file */
export function countExamples (file) {
  return fs.readFileSync(file, 'utf8').split('\n').filter((l) => l.trim()).length
}

/**
 * @param {string} dir
 * @param {number} round
 */
export function roundDir (dir, round) {
  return path.join(dir, 'rounds', String(round))
}

/**
 * @param {string} dir
 * @param {number} round
 */
export function mergedPath (dir, round) {
  return path.join(roundDir(dir, round), 'merged.safetensors')
}

/** @param {State} state */
function names (state) {
  /** @param {string} key */
  return (key) => state.members.get(key) ?? key.slice(0, 8)
}

/**
 * Runs this member's side of the training rounds: train locally, publish the
 * adapter, wait for every member's adapter, merge, and report the merged hash.
 * Picks up where it stopped if it is restarted.
 *
 * @param {{
 *   node: TuneNode,
 *   trainer: Trainer,
 *   data: string,
 *   rounds: number,
 *   dir: string,
 *   log: Log,
 *   signal?: AbortSignal,
 *   lingerMs?: number
 * }} opts
 */
export async function runRounds ({ node, trainer, data, rounds, dir, log, signal, lingerMs = 120000 }) {
  const me = node.identity.publicKey.toString('hex')
  const examples = countExamples(data)
  if (examples === 0) throw new Error(`${data} has no examples`)

  const writer = node.base.local.key.toString('hex')
  if (await node.join()) log('join-requested')
  const state0 = await waitFor(node, (s) => s.team && s.writers.get(writer) === me && s, {
    signal,
    onWait: () => log('waiting', { for: 'team' })
  })
  if (!state0.team) throw new Error('unreachable')
  const config = validateConfig(state0.team.config)
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(config, null, 2) + '\n', { mode: 0o600 })
  const aHash = await trainer.aHash(config)
  if (aHash !== state0.team.aHash) {
    throw new Error('this trainer builds a different frozen A than the team. Check that the model and trainer versions match the owner\'s.')
  }
  log('ready', { team: state0.team.name, members: state0.members.size, examples })

  /** @type {{ round: number, sha256: string }[]} */
  const merged = []
  for (let round = 1; round <= rounds; round++) {
    signal?.throwIfAborted()
    fs.mkdirSync(roundDir(dir, round), { recursive: true, mode: 0o700 })
    const mergedFile = mergedPath(dir, round)
    const prev = round > 1 ? mergedPath(dir, round - 1) : null

    await node.refresh()
    if (!node.state.posts.get(round)?.has(me)) {
      const out = path.join(roundDir(dir, round), 'own.safetensors')
      log('training', { round, examples })
      const res = await trainer.train({ config, data, init: prev, out, round })
      if (res.aHash !== state0.team.aHash) throw new Error('trainer used a different frozen A')
      const bytes = fs.readFileSync(out)
      const stored = await node.putAdapter(bytes)
      await node.append({ type: 'post', round, ...stored, examples: res.examples, aHash: res.aHash, loss: res.loss })
      log('posted', { round, loss: res.loss, seconds: res.seconds, bytes: bytes.length, sha: stored.sha256.slice(0, 12) })
    }

    const posts = await waitFor(node, (s) => roundPosts(s, round), {
      signal,
      onWait: (s) => {
        const name = names(s)
        const missing = [...s.members.keys()].filter((k) => !s.posts.get(round)?.has(k)).map(name)
        log('waiting', { round, for: missing.join(',') })
      }
    })

    const contributions = []
    for (const post of posts) {
      contributions.push({ member: post.member, bytes: await node.getAdapter(post), examples: post.examples })
    }
    const bytes = mergeAdapters(contributions)
    const hash = sha256(bytes)
    fs.writeFileSync(mergedFile, bytes, { mode: 0o600 })
    if (!node.state.merged.get(round)?.has(me)) await node.append({ type: 'merged', round, sha256: hash })
    merged.push({ round, sha256: hash })
    log('merged', { round, members: posts.length, examples: posts.reduce((s, p) => s + p.examples, 0), sha: hash.slice(0, 12) })
  }

  // Stay until every member has reported the last merge, so the others can
  // still fetch this member's adapter, or until the linger time runs out.
  const last = rounds
  try {
    await waitFor(node, (s) => roundAgreement(s, last).reported >= s.members.size, { signal, timeoutMs: lingerMs })
  } catch (err) {
    if (signal?.aborted) throw err
    log('linger-timeout', { round: last })
  }

  await node.refresh()
  const report = merged.map(({ round, sha256: hash }) => {
    const a = roundAgreement(node.state, round)
    return { round, sha256: hash, reported: a.reported, agreed: a.agreed && (a.hashes.length === 0 || a.hashes[0] === hash) }
  })
  for (const r of report) log(r.agreed ? 'agree' : 'DISAGREE', { round: r.round, reported: r.reported, sha: r.sha256.slice(0, 12) })
  return report
}
