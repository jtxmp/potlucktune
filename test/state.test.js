import crypto from 'node:crypto'
import hcrypto from 'hypercore-crypto'
import { describe, expect, it } from 'vitest'
import { decide, emptyState, joinOp, reduce, roundAgreement, roundPosts } from '../src/state.js'

const hex = () => crypto.randomBytes(32).toString('hex')
const baseKey = crypto.randomBytes(32)
const owner = hcrypto.keyPair()
const bob = hcrypto.keyPair()
const ownerHex = owner.publicKey.toString('hex')
const bobHex = bob.publicKey.toString('hex')
const ownerWriter = baseKey.toString('hex')
const bobWriter = crypto.randomBytes(32)
const A_HASH = hex()

function teamOp (extra = {}) {
  return {
    type: 'team',
    name: 't',
    owner: ownerHex,
    members: [{ name: 'alice', key: ownerHex }, { name: 'bob', key: bobHex }],
    config: { seed: 1 },
    aHash: A_HASH,
    ...extra
  }
}

function post (round, extra = {}) {
  return {
    type: 'post',
    round,
    core: hex(),
    blob: { byteOffset: 0, blockOffset: 0, blockLength: 1, byteLength: 100 },
    sha256: hex(),
    examples: 10,
    aHash: A_HASH,
    loss: 0.5,
    ...extra
  }
}

function apply (state, value, from) {
  const result = decide(state, value, from, baseKey)
  for (const e of result.entries) reduce(state, e)
  return result
}

function withTeam () {
  const state = emptyState()
  apply(state, teamOp(), ownerWriter)
  apply(state, joinOp(baseKey, bobWriter, bob), bobWriter.toString('hex'))
  return state
}

describe('decide', () => {
  it('accepts the team only from the bootstrap writer, once', () => {
    const state = emptyState()
    expect(apply(state, teamOp(), hex()).entries).toHaveLength(0)
    expect(apply(state, teamOp(), ownerWriter).entries).toHaveLength(2)
    expect(state.writers.get(ownerWriter)).toBe(ownerHex)
    expect(apply(state, teamOp({ name: 'again' }), ownerWriter).entries).toHaveLength(0)
  })

  it('rejects a team whose owner is not a member', () => {
    const state = emptyState()
    expect(apply(state, teamOp({ owner: hex() }), ownerWriter).entries).toHaveLength(0)
  })

  it('binds a writer to a member only with that member\'s signature', () => {
    const state = emptyState()
    apply(state, teamOp(), ownerWriter)
    const mallory = hcrypto.keyPair()
    // Signed by someone else, claiming to be bob.
    const forged = { ...joinOp(baseKey, bobWriter, mallory), member: bobHex }
    expect(apply(state, forged, bobWriter.toString('hex')).addWriter).toBeUndefined()
    // Signed for a different writer key.
    const replayed = joinOp(baseKey, crypto.randomBytes(32), bob)
    expect(apply(state, replayed, bobWriter.toString('hex')).addWriter).toBeUndefined()
    // A key outside the team.
    expect(apply(state, joinOp(baseKey, bobWriter, mallory), bobWriter.toString('hex')).addWriter).toBeUndefined()

    const ok = apply(state, joinOp(baseKey, bobWriter, bob), bobWriter.toString('hex'))
    expect(ok.addWriter).toBe(bobWriter.toString('hex'))
    expect(state.writers.get(bobWriter.toString('hex'))).toBe(bobHex)

    // A second writer for the same member is refused.
    const other = crypto.randomBytes(32)
    expect(apply(state, joinOp(baseKey, other, bob), other.toString('hex')).addWriter).toBeUndefined()
  })

  it('ignores posts from writers not bound to a member', () => {
    const state = withTeam()
    expect(apply(state, post(1), hex()).entries).toHaveLength(0)
  })

  it('ignores posts with another frozen A, bad rounds or oversized blobs', () => {
    const state = withTeam()
    const w = bobWriter.toString('hex')
    expect(apply(state, post(1, { aHash: hex() }), w).entries).toHaveLength(0)
    expect(apply(state, post(0), w).entries).toHaveLength(0)
    expect(apply(state, post(1.5), w).entries).toHaveLength(0)
    expect(apply(state, post(1, { blob: { byteOffset: 0, blockOffset: 0, blockLength: 1, byteLength: 1e12 } }), w).entries).toHaveLength(0)
    expect(apply(state, post(1, { examples: -1 }), w).entries).toHaveLength(0)
    expect(apply(state, post(1), w).entries).toHaveLength(1)
  })

  it('lets only the owner change the member list', () => {
    const state = withTeam()
    const members = [{ name: 'alice', key: ownerHex }]
    expect(apply(state, { type: 'members', members }, bobWriter.toString('hex')).entries).toHaveLength(0)
    expect(apply(state, { type: 'members', members }, ownerWriter).entries).toHaveLength(1)
    expect(state.members.has(bobHex)).toBe(false)
    // Bob's writer is still bound, but he is no longer a member, so his posts are ignored.
    expect(apply(state, post(1), bobWriter.toString('hex')).entries).toHaveLength(0)
  })
})

describe('rounds', () => {
  it('close only when every member has posted', () => {
    const state = withTeam()
    apply(state, post(1), ownerWriter)
    expect(roundPosts(state, 1)).toBeNull()
    apply(state, post(1), bobWriter.toString('hex'))
    expect(roundPosts(state, 1).map((p) => p.member)).toEqual([ownerHex, bobHex].sort())
  })

  it('pick the same posts whatever order they arrive in', () => {
    const posts = [
      [ownerWriter, post(1, { sha256: 'b'.repeat(64) })],
      [ownerWriter, post(1, { sha256: 'a'.repeat(64) })],
      [bobWriter.toString('hex'), post(1, { sha256: 'c'.repeat(64) })]
    ]
    const pick = (order) => {
      const state = withTeam()
      for (const i of order) apply(state, posts[i][1], posts[i][0])
      return roundPosts(state, 1).map((p) => p.sha256)
    }
    const expected = pick([0, 1, 2])
    expect(pick([2, 1, 0])).toEqual(expected)
    expect(pick([1, 2, 0])).toEqual(expected)
    expect(expected).toContain('a'.repeat(64))
  })

  it('report disagreement between merged hashes', () => {
    const state = withTeam()
    apply(state, { type: 'merged', round: 1, sha256: 'a'.repeat(64) }, ownerWriter)
    expect(roundAgreement(state, 1)).toEqual({ reported: 1, agreed: true, hashes: ['a'.repeat(64)] })
    apply(state, { type: 'merged', round: 1, sha256: 'b'.repeat(64) }, bobWriter.toString('hex'))
    expect(roundAgreement(state, 1).agreed).toBe(false)
  })
})
