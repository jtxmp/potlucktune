// Spike: owner bootstraps an Autobase; member joins with a signed optimistic
// append; both converge on the same view (a plain Hypercore of JSON entries). Also tests Hyperblobs
// replication of a 2 MB blob over the same connection.
import Autobase from 'autobase'
import Corestore from 'corestore'
import Hyperswarm from 'hyperswarm'
import Hyperblobs from 'hyperblobs'
import crypto from 'hypercore-crypto'
import createTestnet from 'hyperdht/testnet.js'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'spike-'))
const testnet = await createTestnet(3)
const bootstrap = testnet.bootstrap

const ownerId = crypto.keyPair()
const memberId = crypto.keyPair()
const topic = crypto.randomBytes(32)

function open (store) {
  return store.get({ name: 'view', valueEncoding: 'json' })
}
async function scan (view) {
  const s = { team: null, writers: {}, posts: {} }
  for (let i = 0; i < view.length; i++) {
    const e = await view.get(i)
    if (e.type === 'team') s.team = e
    else if (e.type === 'writer') s.writers[e.writer] = e.member
    else if (e.type === 'post') s.posts[`${e.round}/${e.member}`] = e
  }
  return s
}

async function apply (nodes, view, host) {
  for (const node of nodes) {
    const v = node.value
    if (!v) continue
    if (v.type === 'team') {
      await view.append(v)
      await view.append({ type: 'writer', writer: node.from.key.toString('hex'), member: v.members[0] })
    } else if (v.type === 'join') {
      const team = (await scan(view)).team
      if (!team) continue
      const id = Buffer.from(v.member, 'hex')
      const msg = Buffer.concat([Buffer.from('join:'), node.from.key])
      if (!team.members.includes(v.member)) continue
      if (!crypto.verify(msg, Buffer.from(v.sig, 'hex'), id)) continue
      await host.ackWriter(node.from.key)
      await host.addWriter(node.from.key, { indexer: false })
      await view.append({ type: 'writer', writer: node.from.key.toString('hex'), member: v.member })
    } else if (v.type === 'post') {
      const member = (await scan(view)).writers[node.from.key.toString('hex')]
      if (!member) continue
      await view.append({ ...v, member })
    }
  }
}

async function makePeer (name, id, baseKey) {
  const store = new Corestore(path.join(tmp, name))
  const base = new Autobase(store, baseKey, { open, apply, optimistic: true, ackInterval: 500, valueEncoding: 'json' })
  await base.ready()
  const swarm = new Hyperswarm({ keyPair: id, bootstrap })
  swarm.on('connection', (conn) => base.replicate(conn))
  swarm.join(topic)
  return { name, store, base, swarm }
}

const t0 = Date.now()
const owner = await makePeer('owner', ownerId, null)
await owner.base.append({ type: 'team', members: [ownerId.publicKey.toString('hex'), memberId.publicKey.toString('hex')] })

const member = await makePeer('member', memberId, owner.base.key)
console.log('member writable before join:', member.base.writable)
const sig = crypto.sign(Buffer.concat([Buffer.from('join:'), member.base.local.key]), memberId.secretKey)
await member.base.append({ type: 'join', member: memberId.publicKey.toString('hex'), sig: sig.toString('hex') }, { optimistic: true })

await new Promise((resolve) => {
  if (member.base.writable) return resolve()
  member.base.once('writable', resolve)
})
console.log('member writable after', Date.now() - t0, 'ms')

// blobs
const blobCore = member.store.get({ name: 'adapters' })
const blobs = new Hyperblobs(blobCore)
await blobs.ready()
const payload = crypto.randomBytes(2 * 1024 * 1024)
const id = await blobs.put(payload)
await member.base.append({ type: 'post', round: 1, core: blobCore.key.toString('hex'), id })

const tPost = Date.now()
console.log('member posted; member view', member.base.view.length, 'member local len', member.base.local.length)
while (true) {
  await owner.base.update()
  const p = (await scan(owner.base.view)).posts['1/' + memberId.publicKey.toString('hex')]
  if (p) break
  await new Promise((r) => setTimeout(r, 100))
}
console.log('owner saw post after', Date.now() - tPost, 'ms')
const post = (await scan(owner.base.view)).posts['1/' + memberId.publicKey.toString('hex')]
const remote = owner.store.get({ key: Buffer.from(post.core, 'hex') })
const rblobs = new Hyperblobs(remote)
const tBlob = Date.now()
const got = await rblobs.get(post.id, { timeout: 30000 })
console.log('blob 2MB fetched in', Date.now() - tBlob, 'ms, equal:', Buffer.compare(got, payload) === 0)

// rejected join: outsider key signs
const outsiderId = crypto.keyPair()
const outsider = await makePeer('outsider', outsiderId, owner.base.key)
const osig = crypto.sign(Buffer.concat([Buffer.from('join:'), outsider.base.local.key]), outsiderId.secretKey)
await outsider.base.append({ type: 'join', member: outsiderId.publicKey.toString('hex'), sig: osig.toString('hex') }, { optimistic: true })
await new Promise((r) => setTimeout(r, 3000))
console.log('outsider writable (expect false):', outsider.base.writable)

await member.base.update()
console.log('view lengths owner/member:', owner.base.view.length, member.base.view.length)

for (const p of [owner, member, outsider]) { await p.swarm.destroy(); await p.base.close(); await p.store.close() }
await testnet.destroy()
console.log('done')
