import { execFileSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { parse, serialize } from '../src/safetensors.js'
import { mergeAdapters, sha256 } from '../src/merge.js'

function randomTensors (seed, shapes = { 'a.lora_B': [6, 3], 'b.lora_B': [4, 3] }) {
  let x = seed
  const next = () => {
    x = (x * 1103515245 + 12345) % 2 ** 31
    return x / 2 ** 31 - 0.5
  }
  return new Map(Object.entries(shapes).map(([name, shape]) => {
    const data = new Float32Array(shape[0] * shape[1]).map(() => next())
    return [name, { shape, data }]
  }))
}

describe('safetensors', () => {
  it('round-trips', () => {
    const t = randomTensors(1)
    const back = parse(serialize(t))
    expect([...back.keys()]).toEqual(['a.lora_B', 'b.lora_B'])
    expect(back.get('a.lora_B').data).toEqual(t.get('a.lora_B').data)
    expect(back.get('a.lora_B').shape).toEqual([6, 3])
  })

  function withHeader (header, bodyLength) {
    const json = Buffer.from(JSON.stringify(header))
    const buf = Buffer.alloc(8 + json.length + bodyLength)
    buf.writeBigUInt64LE(BigInt(json.length))
    json.copy(buf, 8)
    return buf
  }

  it('rejects malformed input', () => {
    expect(() => parse(Buffer.alloc(4))).toThrow()
    const huge = Buffer.alloc(16)
    huge.writeBigUInt64LE(2n ** 40n)
    expect(() => parse(huge)).toThrow(/header length/)
    expect(() => parse(withHeader({ x: { dtype: 'F16', shape: [2], data_offsets: [0, 4] } }, 4))).toThrow(/F32/)
    expect(() => parse(withHeader({ x: { dtype: 'F32', shape: [2], data_offsets: [0, 4] } }, 4))).toThrow(/size/)
    expect(() => parse(withHeader({ x: { dtype: 'F32', shape: [1], data_offsets: [0, 8] } }, 4))).toThrow(/offsets/)
    expect(() => parse(withHeader({ x: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } }, 8))).toThrow(/trailing/)
    expect(() => parse(withHeader({
      x: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
      y: { dtype: 'F32', shape: [1], data_offsets: [0, 4] }
    }, 4))).toThrow(/overlap/)
  })
})

describe('mergeAdapters', () => {
  const contributions = [1, 2, 3].map((s, i) => ({ member: crypto.createHash('sha256').update(String(s)).digest('hex'), examples: 10 + i * 5, bytes: serialize(randomTensors(s)) }))

  it('gives the same bytes whatever order the members are listed in', () => {
    const a = mergeAdapters(contributions)
    const b = mergeAdapters([...contributions].reverse())
    expect(sha256(a)).toBe(sha256(b))
  })

  it('is the example-weighted mean of B', () => {
    const merged = parse(mergeAdapters(contributions)).get('a.lora_B').data
    const parsed = contributions.map((c) => parse(c.bytes).get('a.lora_B').data)
    const total = contributions.reduce((s, c) => s + c.examples, 0)
    for (let i = 0; i < merged.length; i++) {
      const expected = parsed.reduce((s, d, j) => s + d[i] * contributions[j].examples / total, 0)
      expect(merged[i]).toBeCloseTo(expected, 6)
    }
  })

  it('with a shared A, the merged update B·A equals the mean of the members\' updates (FFA-LoRA)', () => {
    // A: rank 3 x in 5, shared by every member.
    const A = Array.from({ length: 3 }, (_, r) => Array.from({ length: 5 }, (_, c) => Math.sin(r * 5 + c)))
    const matmul = (B, rows) => Array.from({ length: rows }, (_, i) => A[0].map((_, c) => B.slice(i * 3, i * 3 + 3).reduce((s, b, r) => s + b * A[r][c], 0)))
    const merged = parse(mergeAdapters(contributions)).get('a.lora_B').data
    const total = contributions.reduce((s, c) => s + c.examples, 0)
    const updates = contributions.map((c) => matmul(parse(c.bytes).get('a.lora_B').data, 6))
    const meanUpdate = updates[0].map((row, i) => row.map((_, c) => updates.reduce((s, u, j) => s + u[i][c] * contributions[j].examples / total, 0)))
    const mergedUpdate = matmul(merged, 6)
    for (let i = 0; i < 6; i++) for (let c = 0; c < 5; c++) expect(mergedUpdate[i][c]).toBeCloseTo(meanUpdate[i][c], 5)
  })

  it('rejects adapters with different tensors or shapes', () => {
    const odd = { member: 'f'.repeat(64), examples: 1, bytes: serialize(randomTensors(9, { 'a.lora_B': [6, 3] })) }
    expect(() => mergeAdapters([...contributions, odd])).toThrow(/different tensors/)
    const wrongShape = { member: 'f'.repeat(64), examples: 1, bytes: serialize(randomTensors(9, { 'a.lora_B': [3, 6], 'b.lora_B': [4, 3] })) }
    expect(() => mergeAdapters([...contributions, wrongShape])).toThrow(/shape/)
    expect(() => mergeAdapters([{ ...contributions[0], examples: 0 }])).toThrow(/example count/)
    const poisoned = randomTensors(4)
    poisoned.get('b.lora_B').data[2] = NaN
    expect(() => mergeAdapters([...contributions, { member: 'e'.repeat(64), examples: 1, bytes: serialize(poisoned) }])).toThrow(/non-finite/)
  })

  const python = path.join(import.meta.dirname, '..', '.venv', 'bin', 'python')
  it.skipIf(!fs.existsSync(python))('matches the merge in eval/run.py byte for byte', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'merge-'))
    const sorted = [...contributions].sort((a, b) => (a.member < b.member ? -1 : 1))
    const files = sorted.map((c, i) => {
      const f = path.join(dir, `${i}.safetensors`)
      fs.writeFileSync(f, c.bytes)
      return f
    })
    const out = path.join(dir, 'py.safetensors')
    const script = 'import sys,json; sys.path.insert(0, sys.argv[1]); import run; run.merge(json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4])'
    execFileSync(python, ['-c', script, path.join(import.meta.dirname, '..', 'eval'), JSON.stringify(files), JSON.stringify(sorted.map((c) => c.examples)), out])
    const py = parse(fs.readFileSync(out))
    const js = parse(mergeAdapters(contributions))
    for (const name of js.keys()) expect(Buffer.from(py.get(name).data.buffer)).toEqual(Buffer.from(js.get(name).data.buffer))
    fs.rmSync(dir, { recursive: true })
  })
})
