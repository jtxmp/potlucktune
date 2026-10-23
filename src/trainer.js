// @ts-check
import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parse, serialize } from './safetensors.js'

export const TRAINER_SCRIPT = fileURLToPath(new URL('../trainer/trainer.py', import.meta.url))

/**
 * @typedef {{ examples: number, loss: number | null, aHash: string, seconds?: number, peakMemMB?: number }} TrainResult
 * @typedef {{ examples: number, accuracy: number, perLabel?: Record<string, number | null>, seconds?: number }} EvalResult
 * @typedef {{
 *   aHash: (config: Record<string, unknown>) => Promise<string>,
 *   train: (opts: { config: Record<string, unknown>, data: string, init: string | null, out: string, round: number }) => Promise<TrainResult>,
 *   evaluate: (opts: { config: Record<string, unknown>, data: string, adapter: string | null }) => Promise<EvalResult>,
 *   exportAdapter: (opts: { config: Record<string, unknown>, adapter: string, out: string }) => Promise<Record<string, unknown>>
 * }} Trainer
 */

/**
 * The real trainer: `trainer/trainer.py` in a Python environment with torch,
 * transformers, peft and safetensors.
 *
 * @param {{ python?: string, device?: string, workDir: string, onStderr?: (line: string) => void }} opts
 * @returns {Trainer}
 */
export function pythonTrainer ({ python = process.env.POTLUCKTUNE_PYTHON || 'python3', device = 'auto', workDir, onStderr }) {
  fs.mkdirSync(workDir, { recursive: true, mode: 0o700 })

  /** @param {Record<string, unknown>} config */
  const writeConfig = (config) => {
    const file = path.join(workDir, `config-${crypto.createHash('sha256').update(JSON.stringify(config)).digest('hex').slice(0, 16)}.json`)
    fs.writeFileSync(file, JSON.stringify(config), { mode: 0o600 })
    return file
  }

  /** @param {string[]} args */
  const run = (args) => new Promise((resolve, reject) => {
    const child = spawn(python, [TRAINER_SCRIPT, ...args, '--device', device], { stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderrTail = ''
    child.stdout.on('data', (d) => { stdout += d })
    child.stderr.on('data', (d) => {
      const text = d.toString()
      stderrTail = (stderrTail + text).slice(-4000)
      if (onStderr) for (const line of text.split('\n')) if (line.trim()) onStderr(line)
    })
    child.on('error', (err) => reject(new Error(`could not start ${python}: ${err.message}. Set --python or POTLUCKTUNE_PYTHON.`)))
    child.on('close', (code) => {
      if (code !== 0) return reject(new Error(`trainer exited with code ${code}: ${stderrTail.trim().split('\n').slice(-3).join(' | ')}`))
      try {
        resolve(JSON.parse(stdout.trim().split('\n').at(-1) ?? ''))
      } catch {
        reject(new Error('trainer printed no result'))
      }
    })
  })

  return {
    async aHash (config) {
      const res = await run(['a-hash', '--config', writeConfig(config)])
      return res.aHash
    },
    async train ({ config, data, init, out, round }) {
      const args = ['train', '--config', writeConfig(config), '--data', data, '--out', out, '--round', String(round)]
      if (init) args.push('--init', init)
      return run(args)
    },
    async evaluate ({ config, data, adapter }) {
      const args = ['eval', '--config', writeConfig(config), '--data', data]
      if (adapter) args.push('--adapter', adapter)
      return run(args)
    },
    async exportAdapter ({ config, adapter, out }) {
      return run(['export', '--config', writeConfig(config), '--adapter', adapter, '--out', out])
    }
  }
}

const FAKE_SHAPES = { 'model.layers.0.self_attn.q_proj.lora_B': [16, 4], 'model.layers.0.self_attn.v_proj.lora_B': [16, 4] }

/**
 * A stand-in trainer for tests and `potlucktune try`: no Python, no model.
 * Each member's B moves toward a target derived from its data, so merges are
 * deterministic and visibly depend on every member.
 *
 * @param {{ delayMs?: number }} [opts]
 * @returns {Trainer}
 */
export function fakeTrainer ({ delayMs = 200 } = {}) {
  /** @param {Record<string, unknown>} config */
  const aHash = (config) => crypto.createHash('sha256').update(`fake-a/${config.seed}`).digest('hex')
  return {
    async aHash (config) { return aHash(config) },
    async train ({ config, data, init, out, round }) {
      const rows = fs.readFileSync(data, 'utf8').split('\n').filter((l) => l.trim())
      const digest = crypto.createHash('sha256').update(rows.join('\n')).digest()
      const start = init ? parse(fs.readFileSync(init)) : null
      /** @type {import('./safetensors.js').Tensors} */
      const tensors = new Map()
      for (const [name, shape] of Object.entries(FAKE_SHAPES)) {
        const count = shape[0] * shape[1]
        const data = new Float32Array(count)
        const prev = start?.get(name)?.data
        for (let i = 0; i < count; i++) {
          const target = (digest[i % digest.length] - 128) / 128
          const from = prev ? prev[i] : 0
          data[i] = from + (target - from) * 0.5
        }
        tensors.set(name, { shape, data })
      }
      await new Promise((resolve) => setTimeout(resolve, delayMs))
      fs.writeFileSync(out, serialize(tensors))
      return { examples: rows.length, loss: Math.round(1000 / (round + 1)) / 1000, aHash: aHash(config) }
    },
    async evaluate ({ data }) {
      const rows = fs.readFileSync(data, 'utf8').split('\n').filter((l) => l.trim())
      return { examples: rows.length, accuracy: 0 }
    },
    async exportAdapter ({ out }) {
      return { out }
    }
  }
}
