// @ts-check
import crypto from 'node:crypto'
import { parse, serialize } from './safetensors.js'

/**
 * @typedef {import('./safetensors.js').Tensors} Tensors
 * @typedef {{ member: string, bytes: Buffer, examples: number }} Contribution
 */

/** @param {Buffer} bytes */
export function sha256 (bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex')
}

/**
 * FFA-LoRA merge: the mean of every member's B, weighted by example count.
 *
 * Every member must get the same bytes from the same inputs, so the order is
 * fixed (sorted by member key), sums are in float64, and the result is cast
 * to float32 once. All adapters must have the same tensor names and shapes.
 *
 * @param {Contribution[]} contributions
 * @returns {Buffer}
 */
export function mergeAdapters (contributions) {
  if (contributions.length === 0) throw new Error('merge: nothing to merge')
  const sorted = [...contributions].sort((a, b) => (a.member < b.member ? -1 : a.member > b.member ? 1 : 0))
  for (const c of sorted) {
    if (!Number.isInteger(c.examples) || c.examples <= 0) throw new Error(`merge: ${c.member.slice(0, 8)}: bad example count`)
  }
  const total = sorted.reduce((sum, c) => sum + c.examples, 0)
  const parsed = sorted.map((c) => parse(c.bytes))

  const names = [...parsed[0].keys()].sort()
  for (let i = 1; i < parsed.length; i++) {
    const other = [...parsed[i].keys()].sort()
    if (other.length !== names.length || other.some((n, j) => n !== names[j])) {
      throw new Error(`merge: ${sorted[i].member.slice(0, 8)}: adapter has different tensors`)
    }
  }

  /** @type {Tensors} */
  const out = new Map()
  for (const name of names) {
    const first = /** @type {import('./safetensors.js').Tensor} */ (parsed[0].get(name))
    const acc = new Float64Array(first.data.length)
    for (let i = 0; i < parsed.length; i++) {
      const t = /** @type {import('./safetensors.js').Tensor} */ (parsed[i].get(name))
      if (t.shape.length !== first.shape.length || t.shape.some((d, j) => d !== first.shape[j])) {
        throw new Error(`merge: ${sorted[i].member.slice(0, 8)}: ${name} has a different shape`)
      }
      const w = sorted[i].examples / total
      for (let j = 0; j < acc.length; j++) {
        // One NaN or Inf would spread to every member's merged model.
        if (!Number.isFinite(t.data[j])) throw new Error(`merge: ${sorted[i].member.slice(0, 8)}: ${name} has a non-finite value`)
        acc[j] += t.data[j] * w
      }
    }
    out.set(name, { shape: first.shape, data: Float32Array.from(acc) })
  }
  return serialize(out)
}
