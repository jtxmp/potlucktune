// @ts-check
import Autobase from 'autobase'
import Corestore from 'corestore'
import Hyperblobs from 'hyperblobs'
import Hyperswarm from 'hyperswarm'
import hcrypto from 'hypercore-crypto'
import { MAX_ADAPTER_BYTES, decide, emptyState, joinOp, reduce, scan } from './state.js'
import { sha256 } from './merge.js'
import { shortKey } from './log.js'

/**
 * @typedef {import('./team.js').Team} Team
 * @typedef {import('./state.js').State} State
 * @typedef {import('./state.js').Post} Post
 * @typedef {(event: string, fields?: Record<string, unknown>) => void} Log
 */

const MIN_LOOKUP_MS = 5000
const MAX_LOOKUP_MS = 60000

/**
 * The swarm topic for a team. Derived from the team file's topic, so
 * PotluckTune peers don't meet PotluckLLM or PotluckRAG peers of the same team.
 *
 * @param {Buffer} teamTopic
 */
export function tuneTopic (teamTopic) {
  return hcrypto.hash([Buffer.from('potlucktune/topic/v1'), teamTopic])
}

/**
 * Opens a member's view of a training run: the shared Autobase (operations
 * and rounds), this member's adapter store, and the swarm connections to the
 * rest of the team.
 *
 * @param {{
 *   storage: string,
 *   seed: Buffer,
 *   team: Team,
 *   baseKey: Buffer | null,
 *   bootstrap?: { host: string, port: number }[],
 *   log: Log
 * }} opts
 */
export async function openNode ({ storage, seed, team, baseKey, bootstrap, log }) {
  const identity = hcrypto.keyPair(seed)
  const store = new Corestore(storage)
  let currentTeam = team

  /** @type {Autobase} */
  const base = new Autobase(store, baseKey, {
    valueEncoding: 'json',
    optimistic: true,
    ackInterval: 1000,
    open: (viewStore) => viewStore.get({ name: 'view', valueEncoding: 'json' }),
    apply: async (nodes, view, host) => {
      const state = await scan(view)
      for (const node of nodes) {
        if (node.value === null) continue
        const from = node.from.key.toString('hex')
        const result = decide(state, node.value, from, base.key)
        for (const entry of result.entries) {
          await view.append(entry)
          reduce(state, entry)
        }
        if (result.addWriter) {
          const key = Buffer.from(result.addWriter, 'hex')
          await host.ackWriter(key)
          await host.addWriter(key, { indexer: false })
        }
      }
    }
  })
  await base.ready()

  const blobCore = store.get({ name: 'adapters' })
  const blobs = new Hyperblobs(blobCore)
  await blobs.ready()

  const swarm = new Hyperswarm({
    seed,
    bootstrap,
    firewall: (remotePublicKey) => {
      const ok = currentTeam.isMember(remotePublicKey) && !remotePublicKey.equals(identity.publicKey)
      if (!ok) log('reject', { peer: shortKey(remotePublicKey) })
      return !ok
    }
  })
  /** @type {Set<string>} */
  const peers = new Set()
  swarm.on('connection', (conn, info) => {
    const peer = info.publicKey.toString('hex')
    peers.add(peer)
    log('connect', { peer: currentTeam.memberName(info.publicKey) ?? shortKey(info.publicKey) })
    conn.on('error', () => {})
    conn.on('close', () => {
      peers.delete(peer)
      log('disconnect', { peer: currentTeam.memberName(info.publicKey) ?? shortKey(info.publicKey) })
    })
    base.replicate(conn)
  })
  const discovery = swarm.join(tuneTopic(team.topic), { server: true, client: true })

  // Hyperswarm repeats a topic lookup only every 10 minutes, so members that
  // start at the same moment may never find each other (PotluckRAG's spike
  // traced this). While any team member is not connected, look up again with
  // backoff.
  let closed = false
  let lookupDelay = MIN_LOOKUP_MS
  /** @type {NodeJS.Timeout | undefined} */
  let lookupTimer
  const lookupAgain = () => {
    if (closed) return
    const self = identity.publicKey.toString('hex')
    if (currentTeam.members.some((m) => m.key !== self && !peers.has(m.key))) {
      discovery.refresh().catch((err) => log('lookup-error', { error: err.message }))
      lookupDelay = Math.min(lookupDelay * 2, MAX_LOOKUP_MS)
    } else {
      lookupDelay = MIN_LOOKUP_MS
    }
    lookupTimer = setTimeout(lookupAgain, lookupDelay)
  }
  lookupTimer = setTimeout(lookupAgain, lookupDelay)

  /** @type {State} */
  let state = emptyState()
  const refresh = async () => {
    state = await scan(base.view)
    return state
  }
  await base.update()
  await refresh()

  return {
    base,
    identity,
    swarm,
    discovery,
    peers,
    get state () { return state },
    refresh,
    /** @param {Team} t */
    setTeam (t) { currentTeam = t },

    /** Appends a signed join request, unless this writer is already bound. */
    async join () {
      await refresh()
      const writer = base.local.key.toString('hex')
      if (state.writers.has(writer)) return false
      await base.append(joinOp(base.key, base.local.key, identity), { optimistic: true })
      return true
    },

    /** @param {Record<string, unknown>} op */
    async append (op) {
      await base.append(op)
    },

    /**
     * Stores an adapter in this member's blob core.
     *
     * @param {Buffer} bytes
     */
    async putAdapter (bytes) {
      if (bytes.length > MAX_ADAPTER_BYTES) throw new Error(`adapter is ${bytes.length} bytes; the limit is ${MAX_ADAPTER_BYTES}`)
      const blob = await blobs.put(bytes)
      return { core: blobCore.key.toString('hex'), blob, sha256: sha256(bytes) }
    },

    /**
     * Fetches a member's adapter and checks it against the hash in their post.
     *
     * @param {Post} post
     * @param {number} [timeout]
     */
    async getAdapter (post, timeout = 120000) {
      if (post.blob.byteLength > MAX_ADAPTER_BYTES) throw new Error('adapter too large')
      const core = post.core === blobCore.key.toString('hex') ? blobCore : store.get({ key: Buffer.from(post.core, 'hex') })
      const remote = new Hyperblobs(core)
      try {
        const bytes = await remote.get(post.blob, { timeout })
        if (!bytes) throw new Error('adapter not available')
        if (bytes.length !== post.blob.byteLength || sha256(bytes) !== post.sha256) {
          throw new Error(`adapter from ${post.member.slice(0, 8)} does not match its hash`)
        }
        return bytes
      } finally {
        if (core !== blobCore) await core.close()
      }
    },

    async close () {
      closed = true
      clearTimeout(lookupTimer)
      await swarm.destroy()
      await base.close()
      await blobCore.close()
      await store.close()
    }
  }
}

/** @typedef {Awaited<ReturnType<typeof openNode>>} TuneNode */
