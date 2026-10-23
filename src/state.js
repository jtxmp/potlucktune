// @ts-check
import crypto from 'hypercore-crypto'

// The shared state of a training run. Writers append operations to the
// Autobase; `decide` validates each one against the state so far and returns
// the entries to append to the view (and a writer to add, for joins). The view
// holds only validated entries, and `reduce` rebuilds the state from them.
//
// Round membership is order-independent: a round closes when every member has
// a post for it, and a member's post for a round is the one with the smallest
// hash if they somehow posted twice. So members agree on which adapters go
// into a merge no matter in which order the posts arrived.

export const MAX_ADAPTER_BYTES = 64 * 1024 * 1024
export const MAX_ROUND = 10000
const HEX32 = /^[0-9a-f]{64}$/
const JOIN_PREFIX = Buffer.from('potlucktune/join/v1\0')

/**
 * @typedef {{ name: string, key: string }} Member
 * @typedef {{ byteOffset: number, blockOffset: number, blockLength: number, byteLength: number }} BlobId
 * @typedef {{ type: 'post', member: string, round: number, core: string, blob: BlobId, sha256: string, examples: number, aHash: string, loss: number | null }} Post
 * @typedef {{
 *   team: { name: string, owner: string, config: Record<string, unknown>, aHash: string } | null,
 *   members: Map<string, string>,
 *   writers: Map<string, string>,
 *   posts: Map<number, Map<string, Post>>,
 *   merged: Map<number, Map<string, string>>
 * }} State
 */

/** @returns {State} */
export function emptyState () {
  return { team: null, members: new Map(), writers: new Map(), posts: new Map(), merged: new Map() }
}

/**
 * Message a member signs to bind their Autobase writer to their identity.
 *
 * @param {Buffer} baseKey
 * @param {Buffer} writerKey
 */
export function joinMessage (baseKey, writerKey) {
  return Buffer.concat([JOIN_PREFIX, baseKey, writerKey])
}

/**
 * @param {Buffer} baseKey
 * @param {Buffer} writerKey
 * @param {{ publicKey: Buffer, secretKey: Buffer }} identity
 */
export function joinOp (baseKey, writerKey, identity) {
  return {
    type: 'join',
    member: identity.publicKey.toString('hex'),
    sig: crypto.sign(joinMessage(baseKey, writerKey), identity.secretKey).toString('hex')
  }
}

/** @param {unknown} v */
const isHex32 = (v) => typeof v === 'string' && HEX32.test(v)

/** @param {unknown} members @returns {Member[] | null} */
function validMembers (members) {
  if (!Array.isArray(members) || members.length === 0 || members.length > 256) return null
  const keys = new Set()
  for (const m of members) {
    if (!m || typeof m.name !== 'string' || m.name.length > 64 || !isHex32(m.key) || keys.has(m.key)) return null
    keys.add(m.key)
  }
  return members.map((m) => ({ name: m.name, key: m.key }))
}

/** @param {unknown} id @returns {id is BlobId} */
function validBlobId (id) {
  if (!id || typeof id !== 'object') return false
  const b = /** @type {Record<string, unknown>} */ (id)
  return ['byteOffset', 'blockOffset', 'blockLength', 'byteLength'].every((k) => Number.isSafeInteger(b[k]) && /** @type {number} */ (b[k]) >= 0) &&
    /** @type {number} */ (b.byteLength) > 0 && /** @type {number} */ (b.byteLength) <= MAX_ADAPTER_BYTES
}

/** @param {unknown} r */
const validRound = (r) => Number.isInteger(r) && /** @type {number} */ (r) >= 1 && /** @type {number} */ (r) <= MAX_ROUND

/**
 * @param {State} state
 * @param {any} value the operation a writer appended
 * @param {string} from hex key of the writer core that appended it
 * @param {Buffer} baseKey
 * @returns {{ entries: Record<string, unknown>[], addWriter?: string }}
 */
export function decide (state, value, from, baseKey) {
  const none = { entries: [] }
  if (!value || typeof value !== 'object') return none
  const member = state.writers.get(from)

  if (value.type === 'team') {
    // Only the first operation of the base can create the team, and only the
    // bootstrap writer can append before anyone else has joined.
    if (state.team || from !== baseKey.toString('hex')) return none
    const members = validMembers(value.members)
    if (!members || typeof value.name !== 'string' || !isHex32(value.owner) || !isHex32(value.aHash)) return none
    if (!members.some((m) => m.key === value.owner)) return none
    if (!value.config || typeof value.config !== 'object' || Array.isArray(value.config)) return none
    return {
      entries: [
        { type: 'team', name: value.name.slice(0, 64), owner: value.owner, members, config: value.config, aHash: value.aHash },
        { type: 'writer', writer: from, member: value.owner }
      ]
    }
  }

  if (!state.team) return none

  if (value.type === 'join') {
    if (member || !isHex32(value.member) || typeof value.sig !== 'string' || !/^[0-9a-f]{128}$/.test(value.sig)) return none
    if (!state.members.has(value.member)) return none
    if ([...state.writers.values()].includes(value.member)) return none
    const ok = crypto.verify(joinMessage(baseKey, Buffer.from(from, 'hex')), Buffer.from(value.sig, 'hex'), Buffer.from(value.member, 'hex'))
    if (!ok) return none
    return { entries: [{ type: 'writer', writer: from, member: value.member }], addWriter: from }
  }

  // Everything below needs a writer bound to a current member.
  if (!member || !state.members.has(member)) return none

  if (value.type === 'members') {
    if (member !== state.team.owner) return none
    const members = validMembers(value.members)
    if (!members || !members.some((m) => m.key === state.team.owner)) return none
    return { entries: [{ type: 'members', members }] }
  }

  if (value.type === 'post') {
    if (!validRound(value.round) || !isHex32(value.core) || !validBlobId(value.blob) || !isHex32(value.sha256)) return none
    if (!Number.isInteger(value.examples) || value.examples <= 0 || value.examples > 1e9) return none
    if (value.aHash !== state.team.aHash) return none
    const loss = typeof value.loss === 'number' && Number.isFinite(value.loss) ? value.loss : null
    const { byteOffset, blockOffset, blockLength, byteLength } = value.blob
    return {
      entries: [{
        type: 'post',
        member,
        round: value.round,
        core: value.core,
        blob: { byteOffset, blockOffset, blockLength, byteLength },
        sha256: value.sha256,
        examples: value.examples,
        aHash: value.aHash,
        loss
      }]
    }
  }

  if (value.type === 'merged') {
    if (!validRound(value.round) || !isHex32(value.sha256)) return none
    return { entries: [{ type: 'merged', member, round: value.round, sha256: value.sha256 }] }
  }

  return none
}

/**
 * Applies one validated view entry to the state.
 *
 * @param {State} state
 * @param {any} entry
 */
export function reduce (state, entry) {
  switch (entry.type) {
    case 'team':
      state.team = { name: entry.name, owner: entry.owner, config: entry.config, aHash: entry.aHash }
      state.members = new Map(entry.members.map((/** @type {Member} */ m) => [m.key, m.name]))
      break
    case 'members':
      state.members = new Map(entry.members.map((/** @type {Member} */ m) => [m.key, m.name]))
      break
    case 'writer':
      state.writers.set(entry.writer, entry.member)
      break
    case 'post': {
      let round = state.posts.get(entry.round)
      if (!round) state.posts.set(entry.round, (round = new Map()))
      const prev = round.get(entry.member)
      if (!prev || entry.sha256 < prev.sha256) round.set(entry.member, entry)
      break
    }
    case 'merged': {
      let round = state.merged.get(entry.round)
      if (!round) state.merged.set(entry.round, (round = new Map()))
      if (!round.has(entry.member)) round.set(entry.member, entry.sha256)
      break
    }
  }
  return state
}

/**
 * The posts that make up a round, sorted by member key, or null while some
 * current member has not posted.
 *
 * @param {State} state
 * @param {number} round
 * @returns {Post[] | null}
 */
export function roundPosts (state, round) {
  const posts = state.posts.get(round)
  if (!posts || state.members.size === 0) return null
  const keys = [...state.members.keys()].sort()
  if (!keys.every((k) => posts.has(k))) return null
  return keys.map((k) => /** @type {Post} */ (posts.get(k)))
}

/**
 * Whether every member who reported a merge for this round got the same hash.
 *
 * @param {State} state
 * @param {number} round
 * @returns {{ reported: number, agreed: boolean, hashes: string[] }}
 */
export function roundAgreement (state, round) {
  const merged = state.merged.get(round) ?? new Map()
  const hashes = [...new Set(merged.values())]
  return { reported: merged.size, agreed: hashes.length <= 1, hashes }
}

/**
 * Reads every entry of a view core and rebuilds the state.
 *
 * @param {{ length: number, get: (i: number) => Promise<any> }} view
 * @param {State} [state]
 * @param {number} [from]
 */
export async function scan (view, state = emptyState(), from = 0) {
  for (let i = from; i < view.length; i++) reduce(state, await view.get(i))
  return state
}
