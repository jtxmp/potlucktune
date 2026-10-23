import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import createTestnet from 'hyperdht/testnet.js'
import Hyperswarm from 'hyperswarm'
import { openNode, tuneTopic } from '../src/node.js'
import { createRun, mergedPath, runRounds } from '../src/run.js'
import { parseTeam } from '../src/team.js'
import { publicKeyFromSeed } from '../src/identity.js'
import { fakeTrainer } from '../src/trainer.js'
import { DEFAULT_CONFIG } from '../src/config.js'
import { mergeAdapters, sha256 } from '../src/merge.js'

const quiet = () => {}

let testnet
let tmp
beforeAll(async () => {
  testnet = await createTestnet(3)
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'potlucktune-test-'))
})
afterAll(async () => {
  await testnet.destroy()
  fs.rmSync(tmp, { recursive: true, force: true })
})

function member (name) {
  const seed = crypto.randomBytes(32)
  return { name, seed, key: publicKeyFromSeed(seed).toString('hex') }
}

function writeData (name, lines) {
  const file = path.join(tmp, `${name}-${crypto.randomBytes(4).toString('hex')}.jsonl`)
  fs.writeFileSync(file, lines.map((t) => JSON.stringify({ text: t, label: 'x' })).join('\n') + '\n')
  return file
}

describe('three members, fake trainer', () => {
  it('trains two rounds and every member gets the same merged adapter', async () => {
    const people = [member('alice'), member('bob'), member('carol')]
    const team = parseTeam({ name: 't', topic: crypto.randomBytes(32).toString('hex'), members: people.map(({ name, key }) => ({ name, key })) })
    const trainer = fakeTrainer({ delayMs: 50 })

    const owner = await openNode({ storage: path.join(tmp, 'alice'), seed: people[0].seed, team, baseKey: null, bootstrap: testnet.bootstrap, log: quiet })
    await createRun({ node: owner, trainer, name: 't', members: team.members, config: { ...DEFAULT_CONFIG, labels: ['x', 'y'], seed: 1 } })
    const others = await Promise.all(people.slice(1).map((p) => openNode({
      storage: path.join(tmp, p.name), seed: p.seed, team, baseKey: owner.base.key, bootstrap: testnet.bootstrap, log: quiet
    })))
    const nodes = [owner, ...others]
    const data = people.map((p, i) => writeData(p.name, Array.from({ length: 10 + i }, (_, j) => `${p.name} ${j}`)))

    const reports = await Promise.all(nodes.map((node, i) => runRounds({
      node, trainer, data: data[i], rounds: 2, dir: path.join(tmp, `${people[i].name}-run`), log: quiet, lingerMs: 20000
    })))

    for (const report of reports) {
      expect(report.map((r) => r.agreed)).toEqual([true, true])
      expect(report.map((r) => r.reported)).toEqual([3, 3])
    }
    const hashes = reports.map((r) => r[1].sha256)
    expect(new Set(hashes).size).toBe(1)

    // The merge equals one computed directly from the three round-2 adapters.
    const state = await owner.refresh()
    const posts = [...state.posts.get(2).values()]
    const contributions = await Promise.all(posts.map(async (p) => ({ member: p.member, examples: p.examples, bytes: await owner.getAdapter(p) })))
    expect(sha256(mergeAdapters(contributions))).toBe(hashes[0])
    expect(sha256(fs.readFileSync(mergedPath(path.join(tmp, 'bob-run'), 2)))).toBe(hashes[0])
    expect(contributions.map((c) => c.examples).sort()).toEqual([10, 11, 12])

    await Promise.all(nodes.map((n) => n.close()))
  }, 60000)

  it('refuses connections from keys outside the team and ignores their joins', async () => {
    const people = [member('alice'), member('bob')]
    const outsider = member('mallory')
    const team = parseTeam({ name: 't2', topic: crypto.randomBytes(32).toString('hex'), members: people.map(({ name, key }) => ({ name, key })) })
    const trainer = fakeTrainer({ delayMs: 10 })
    const rejected = []
    const owner = await openNode({
      storage: path.join(tmp, 'o2'), seed: people[0].seed, team, baseKey: null, bootstrap: testnet.bootstrap,
      log: (event, fields) => { if (event === 'reject') rejected.push(fields.peer) }
    })
    await createRun({ node: owner, trainer, name: 't2', members: team.members, config: { ...DEFAULT_CONFIG, labels: ['x', 'y'], seed: 2 } })

    // The outsider uses a team file that lists itself, so its own firewall lets it dial.
    const outsiderTeam = parseTeam({ name: team.name, topic: team.topic.toString('hex'), members: [...team.members, { name: 'mallory', key: outsider.key }] })
    const intruder = await openNode({ storage: path.join(tmp, 'm2'), seed: outsider.seed, team: outsiderTeam, baseKey: owner.base.key, bootstrap: testnet.bootstrap, log: quiet })
    await intruder.join()
    await intruder.discovery.flushed()
    await new Promise((resolve) => setTimeout(resolve, 3000))

    expect(rejected).toContain(outsider.key.slice(0, 8))
    expect(owner.peers.size).toBe(0)
    const state = await owner.refresh()
    expect([...state.writers.values()]).toEqual([people[0].key])

    await intruder.close()
    await owner.close()
  }, 30000)
})

describe('topic', () => {
  it('differs from the team topic, so other Potluck tools do not connect', () => {
    const topic = crypto.randomBytes(32)
    expect(tuneTopic(topic).equals(topic)).toBe(false)
    expect(tuneTopic(topic)).toHaveLength(32)
  })

  it('identity public key matches the key Hyperswarm uses for the same seed', async () => {
    const seed = crypto.randomBytes(32)
    const swarm = new Hyperswarm({ seed, bootstrap: testnet.bootstrap })
    expect(swarm.keyPair.publicKey.equals(publicKeyFromSeed(seed))).toBe(true)
    await swarm.destroy()
  })
})
