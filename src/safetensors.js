// @ts-check

// Strict reader and writer for the subset of safetensors that adapters use:
// little-endian float32 tensors, no metadata beyond `__metadata__`. Bytes from
// peers go through `parse`, so it checks every length and offset.

const MAX_HEADER = 1024 * 1024

/**
 * @typedef {{ shape: number[], data: Float32Array }} Tensor
 * @typedef {Map<string, Tensor>} Tensors
 */

/**
 * @param {Buffer} buf
 * @returns {Tensors}
 */
export function parse (buf) {
  if (buf.length < 8) throw new Error('safetensors: file too short')
  const headerLen = Number(buf.readBigUInt64LE(0))
  if (headerLen > MAX_HEADER || 8 + headerLen > buf.length) throw new Error('safetensors: bad header length')

  let header
  try {
    header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8'))
  } catch {
    throw new Error('safetensors: header is not JSON')
  }
  if (!header || typeof header !== 'object' || Array.isArray(header)) throw new Error('safetensors: bad header')

  const body = buf.subarray(8 + headerLen)
  /** @type {Tensors} */
  const tensors = new Map()
  const ranges = []
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue
    if (info?.dtype !== 'F32') throw new Error(`safetensors: ${name}: only F32 is supported`)
    const shape = info.shape
    if (!Array.isArray(shape) || !shape.every((d) => Number.isInteger(d) && d >= 0)) {
      throw new Error(`safetensors: ${name}: bad shape`)
    }
    const [start, end] = info.data_offsets ?? []
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > body.length) {
      throw new Error(`safetensors: ${name}: bad offsets`)
    }
    const count = shape.reduce((a, b) => a * b, 1)
    if (end - start !== count * 4) throw new Error(`safetensors: ${name}: size does not match shape`)
    ranges.push([start, end])
    // Copy so the tensor is aligned and independent of the input buffer.
    const data = new Float32Array(count)
    const view = new DataView(body.buffer, body.byteOffset + start, end - start)
    for (let i = 0; i < count; i++) data[i] = view.getFloat32(i * 4, true)
    tensors.set(name, { shape, data })
  }

  ranges.sort((a, b) => a[0] - b[0])
  let pos = 0
  for (const [start, end] of ranges) {
    if (start !== pos) throw new Error('safetensors: tensors overlap or leave holes')
    pos = end
  }
  if (pos !== body.length) throw new Error('safetensors: trailing bytes')
  return tensors
}

/**
 * Serializes with tensors in sorted name order and a header with sorted keys,
 * so equal tensors always give equal bytes.
 *
 * @param {Tensors} tensors
 * @returns {Buffer}
 */
export function serialize (tensors) {
  const names = [...tensors.keys()].sort()
  /** @type {Record<string, unknown>} */
  const header = {}
  let offset = 0
  for (const name of names) {
    const t = /** @type {Tensor} */ (tensors.get(name))
    const size = t.data.length * 4
    header[name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + size] }
    offset += size
  }
  let json = JSON.stringify(header)
  // Pad the header with spaces to a multiple of 8, as the reference writer does.
  json += ' '.repeat((8 - ((8 + Buffer.byteLength(json)) % 8)) % 8)
  const headerBuf = Buffer.from(json, 'utf8')
  const out = Buffer.alloc(8 + headerBuf.length + offset)
  out.writeBigUInt64LE(BigInt(headerBuf.length), 0)
  headerBuf.copy(out, 8)
  let pos = 8 + headerBuf.length
  for (const name of names) {
    const t = /** @type {Tensor} */ (tensors.get(name))
    for (let i = 0; i < t.data.length; i++) {
      out.writeFloatLE(t.data[i], pos)
      pos += 4
    }
  }
  return out
}
