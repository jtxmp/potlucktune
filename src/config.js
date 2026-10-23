// @ts-check

// The training config is set by the team owner and travels to every member
// inside the shared base. Members run it through `validateConfig` before
// handing it to the trainer, so a config can only name a Hugging Face model
// id (not a local path) and only set values in sane ranges.

export const DEFAULT_CONFIG = {
  model: 'Qwen/Qwen2.5-0.5B-Instruct',
  rank: 8,
  alpha: 16,
  targets: ['q_proj', 'v_proj'],
  seed: 1234,
  lr: 0.002,
  steps: 40,
  batch: 8,
  maxLen: 96
}

const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/
const MODULE = /^[A-Za-z0-9_]{1,64}$/

/**
 * @param {unknown} value
 * @param {number} min
 * @param {number} max
 * @param {boolean} [integer]
 */
function inRange (value, min, max, integer = true) {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max && (!integer || Number.isInteger(value))
}

/**
 * @param {any} config
 * @returns {Record<string, unknown>} the config, if valid
 */
export function validateConfig (config) {
  const fail = (/** @type {string} */ msg) => { throw new Error(`training config: ${msg}`) }
  if (!config || typeof config !== 'object' || Array.isArray(config)) fail('not an object')
  if (typeof config.model !== 'string' || !MODEL_ID.test(config.model) || config.model.includes('..')) fail('"model" must be a Hugging Face model id like org/name')
  if (!inRange(config.rank, 1, 256)) fail('"rank" must be an integer from 1 to 256')
  if (!inRange(config.alpha, 0, 1024, false)) fail('"alpha" must be a number from 0 to 1024')
  if (!Array.isArray(config.targets) || config.targets.length === 0 || config.targets.length > 16 || !config.targets.every((t) => typeof t === 'string' && MODULE.test(t))) fail('"targets" must list module names')
  if (!inRange(config.seed, 0, 2 ** 31 - 1)) fail('"seed" must be a non-negative integer')
  if (!inRange(config.lr, 1e-7, 1, false)) fail('"lr" must be between 1e-7 and 1')
  if (!inRange(config.steps, 1, 100000)) fail('"steps" must be an integer from 1 to 100000')
  if (!inRange(config.batch, 1, 1024)) fail('"batch" must be an integer from 1 to 1024')
  if (config.maxLen !== undefined && !inRange(config.maxLen, 8, 4096)) fail('"maxLen" must be an integer from 8 to 4096')
  if (!Array.isArray(config.labels) || config.labels.length < 2 || config.labels.length > 1000) fail('"labels" must list 2 to 1000 labels')
  if (!config.labels.every((/** @type {unknown} */ l) => typeof l === 'string' && l.length > 0 && l.length <= 100 && !l.includes('\n'))) fail('each label must be a short single-line string')
  if (new Set(config.labels).size !== config.labels.length) fail('labels must be unique')
  return config
}

/**
 * Labels found in a JSONL data file, sorted.
 *
 * @param {string} text contents of a JSONL file with { text, label } rows
 */
export function labelsFrom (text) {
  const labels = new Set()
  for (const [i, line] of text.split('\n').entries()) {
    if (!line.trim()) continue
    let row
    try {
      row = JSON.parse(line)
    } catch {
      throw new Error(`line ${i + 1} is not JSON`)
    }
    if (typeof row.label !== 'string') throw new Error(`line ${i + 1} has no string "label"`)
    labels.add(row.label)
  }
  return [...labels].sort()
}
