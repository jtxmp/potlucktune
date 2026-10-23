import { describe, expect, it } from 'vitest'
import { DEFAULT_CONFIG, labelsFrom, validateConfig } from '../src/config.js'

const good = { ...DEFAULT_CONFIG, labels: ['a', 'b'] }

describe('validateConfig', () => {
  it('accepts the default config with labels', () => {
    expect(validateConfig(good)).toBe(good)
  })

  it('only accepts Hugging Face model ids, not paths', () => {
    for (const model of ['/etc/passwd', '../models/x', 'org/../x', 'org', './org/name', 'org/name/extra', 'https://example.com/m']) {
      expect(() => validateConfig({ ...good, model })).toThrow(/model/)
    }
  })

  it('rejects out-of-range values', () => {
    expect(() => validateConfig({ ...good, rank: 0 })).toThrow(/rank/)
    expect(() => validateConfig({ ...good, steps: 1e9 })).toThrow(/steps/)
    expect(() => validateConfig({ ...good, lr: 5 })).toThrow(/lr/)
    expect(() => validateConfig({ ...good, targets: ['q_proj; rm -rf'] })).toThrow(/targets/)
    expect(() => validateConfig({ ...good, labels: ['a'] })).toThrow(/labels/)
    expect(() => validateConfig({ ...good, labels: ['a', 'a'] })).toThrow(/unique/)
    expect(() => validateConfig({ ...good, labels: ['a', 'b\nc'] })).toThrow(/label/)
  })
})

describe('labelsFrom', () => {
  it('collects sorted unique labels', () => {
    expect(labelsFrom('{"text":"x","label":"b"}\n{"text":"y","label":"a"}\n\n{"text":"z","label":"b"}\n')).toEqual(['a', 'b'])
  })

  it('names the bad line', () => {
    expect(() => labelsFrom('{"text":"x","label":"b"}\nnot json\n')).toThrow(/line 2/)
  })
})
