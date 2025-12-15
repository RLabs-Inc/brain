/**
 * Neuromodulation System - Global and Local Modulatory Signals
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - ALL computation on GPU - NEVER convert to JS
 * - Direct mutation of state arrays
 *
 * Implements four major neuromodulatory systems:
 * - Dopamine (DA): Reward prediction error, gates reward-based learning
 * - Serotonin (5-HT): Satiation/mood, reduces plasticity when satisfied
 * - Norepinephrine (NE): Attention/arousal, enhances plasticity during alert states
 * - Acetylcholine (ACh): Learning vs recall mode, enables memory formation
 *
 * Each modulator gates plasticity differently - this is the "third factor"
 * in three-factor learning (pre-spike, post-spike, neuromodulator).
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'

// ============================================================================
// NEUROMODULATOR TYPES
// ============================================================================

export type Neuromodulator =
  | 'dopamine'
  | 'serotonin'
  | 'norepinephrine'
  | 'acetylcholine'

// Short aliases for convenience
export type ModulatorShort = 'DA' | '5HT' | 'NE' | 'ACh'

export const ModulatorMap: Record<ModulatorShort, Neuromodulator> = {
  DA: 'dopamine',
  '5HT': 'serotonin',
  NE: 'norepinephrine',
  ACh: 'acetylcholine',
}

// ============================================================================
// MODULATOR DEFAULTS (biologically inspired)
// ============================================================================

export const ModulatorDefaults = {
  dopamine: {
    baseline: 0, // No tonic dopamine (phasic only)
    decay: 0.9, // Fast decay - reward signals are brief
    min: -1, // Can go negative (worse than expected)
    max: 1, // Bounded
  },
  serotonin: {
    baseline: 0.5, // Moderate tonic level
    decay: 0.99, // Slow decay - mood is stable
    min: 0, // Can't go negative
    max: 1, // Bounded
  },
  norepinephrine: {
    baseline: 0.3, // Low tonic level
    decay: 0.95, // Medium decay
    min: 0, // Can't go negative
    max: 1, // Bounded
  },
  acetylcholine: {
    baseline: 0.5, // Moderate tonic level
    decay: 0.98, // Medium-slow decay
    min: 0, // Can't go negative
    max: 1, // Bounded
  },
} as const

// ============================================================================
// MODULATION REGISTRY (per network)
// ============================================================================

export const registry = $state({
  idToIndex: new SvelteMap<string, number>(),
  indexToId: new SvelteMap<number, string>(),
  allocatedIndices: new Set<number>(),
  freeIndices: [] as number[],
  nextIndex: 0,
})

// ============================================================================
// MODULATION STATE (DIRECT EXPORTS)
// Per network - all GPU scalars for GPU operations
// ============================================================================

// Dopamine - reward prediction error
export const dopamine = $state<ReturnType<typeof mx.array>[]>([])
export const dopamineDecay = $state<ReturnType<typeof mx.array>[]>([])
export const dopamineBaseline = $state<ReturnType<typeof mx.array>[]>([])

// Serotonin - satiation/mood
export const serotonin = $state<ReturnType<typeof mx.array>[]>([])
export const serotoninDecay = $state<ReturnType<typeof mx.array>[]>([])
export const serotoninBaseline = $state<ReturnType<typeof mx.array>[]>([])

// Norepinephrine - attention/arousal
export const norepinephrine = $state<ReturnType<typeof mx.array>[]>([])
export const norepinephrineDecay = $state<ReturnType<typeof mx.array>[]>([])
export const norepinephrineBaseline = $state<ReturnType<typeof mx.array>[]>([])

// Acetylcholine - learning/memory mode
export const acetylcholine = $state<ReturnType<typeof mx.array>[]>([])
export const acetylcholineDecay = $state<ReturnType<typeof mx.array>[]>([])
export const acetylcholineBaseline = $state<ReturnType<typeof mx.array>[]>([])

// ============================================================================
// MODULATION SYSTEM MANAGEMENT
// ============================================================================

/**
 * Options for initializing a modulation system.
 */
export interface ModulationOptions {
  dopamineBaseline?: number
  dopamineDecay?: number
  serotoninBaseline?: number
  serotoninDecay?: number
  norepinephrineBaseline?: number
  norepinephrineDecay?: number
  acetylcholineBaseline?: number
  acetylcholineDecay?: number
}

/**
 * Allocate a modulation system for a network.
 * Returns the modulation index (should match network index).
 */
export function allocateModulation(
  id: string,
  options: ModulationOptions = {}
): number {
  const existing = registry.idToIndex.get(id)
  if (existing !== undefined) return existing

  let index: number
  if (registry.freeIndices.length > 0) {
    index = registry.freeIndices.pop()!
  } else {
    index = registry.nextIndex++
  }

  registry.idToIndex.set(id, index)
  registry.indexToId.set(index, id)
  registry.allocatedIndices.add(index)

  // Initialize dopamine
  const daDefaults = ModulatorDefaults.dopamine
  dopamineBaseline[index] = mx.array(
    options.dopamineBaseline ?? daDefaults.baseline,
    mx.float32
  )
  dopamineDecay[index] = mx.array(
    options.dopamineDecay ?? daDefaults.decay,
    mx.float32
  )
  dopamine[index] = mx.array(
    options.dopamineBaseline ?? daDefaults.baseline,
    mx.float32
  )

  // Initialize serotonin
  const serDefaults = ModulatorDefaults.serotonin
  serotoninBaseline[index] = mx.array(
    options.serotoninBaseline ?? serDefaults.baseline,
    mx.float32
  )
  serotoninDecay[index] = mx.array(
    options.serotoninDecay ?? serDefaults.decay,
    mx.float32
  )
  serotonin[index] = mx.array(
    options.serotoninBaseline ?? serDefaults.baseline,
    mx.float32
  )

  // Initialize norepinephrine
  const neDefaults = ModulatorDefaults.norepinephrine
  norepinephrineBaseline[index] = mx.array(
    options.norepinephrineBaseline ?? neDefaults.baseline,
    mx.float32
  )
  norepinephrineDecay[index] = mx.array(
    options.norepinephrineDecay ?? neDefaults.decay,
    mx.float32
  )
  norepinephrine[index] = mx.array(
    options.norepinephrineBaseline ?? neDefaults.baseline,
    mx.float32
  )

  // Initialize acetylcholine
  const achDefaults = ModulatorDefaults.acetylcholine
  acetylcholineBaseline[index] = mx.array(
    options.acetylcholineBaseline ?? achDefaults.baseline,
    mx.float32
  )
  acetylcholineDecay[index] = mx.array(
    options.acetylcholineDecay ?? achDefaults.decay,
    mx.float32
  )
  acetylcholine[index] = mx.array(
    options.acetylcholineBaseline ?? achDefaults.baseline,
    mx.float32
  )

  return index
}

/**
 * Get modulation index by id.
 */
export function getModulationIndex(id: string): number | undefined {
  return registry.idToIndex.get(id)
}

/**
 * Release a modulation system.
 */
export function releaseModulation(id: string): void {
  const index = registry.idToIndex.get(id)
  if (index === undefined) return

  registry.idToIndex.delete(id)
  registry.indexToId.delete(index)
  registry.allocatedIndices.delete(index)
  registry.freeIndices.push(index)

  // Reset to defaults
  dopamine[index] = mx.array(0, mx.float32)
  dopamineDecay[index] = mx.array(0.9, mx.float32)
  dopamineBaseline[index] = mx.array(0, mx.float32)

  serotonin[index] = mx.array(0.5, mx.float32)
  serotoninDecay[index] = mx.array(0.99, mx.float32)
  serotoninBaseline[index] = mx.array(0.5, mx.float32)

  norepinephrine[index] = mx.array(0.3, mx.float32)
  norepinephrineDecay[index] = mx.array(0.95, mx.float32)
  norepinephrineBaseline[index] = mx.array(0.3, mx.float32)

  acetylcholine[index] = mx.array(0.5, mx.float32)
  acetylcholineDecay[index] = mx.array(0.98, mx.float32)
  acetylcholineBaseline[index] = mx.array(0.5, mx.float32)
}

// ============================================================================
// MODULATOR OPERATIONS (ALL GPU)
// ============================================================================

/**
 * Release a specific neuromodulator.
 * Adds to current level (can accumulate).
 * ALL GPU operations.
 */
export function releaseModulator(
  modIndex: number,
  modulator: Neuromodulator,
  amount: number
) {
  const amountGPU = mx.array(amount, mx.float32)
  const defaults = ModulatorDefaults[modulator]

  switch (modulator) {
    case 'dopamine':
      dopamine[modIndex] = mx.clip(
        mx.add(dopamine[modIndex], amountGPU),
        mx.array(defaults.min),
        mx.array(defaults.max)
      )
      break
    case 'serotonin':
      serotonin[modIndex] = mx.clip(
        mx.add(serotonin[modIndex], amountGPU),
        mx.array(defaults.min),
        mx.array(defaults.max)
      )
      break
    case 'norepinephrine':
      norepinephrine[modIndex] = mx.clip(
        mx.add(norepinephrine[modIndex], amountGPU),
        mx.array(defaults.min),
        mx.array(defaults.max)
      )
      break
    case 'acetylcholine':
      acetylcholine[modIndex] = mx.clip(
        mx.add(acetylcholine[modIndex], amountGPU),
        mx.array(defaults.min),
        mx.array(defaults.max)
      )
      break
  }
}

/**
 * Release modulator by short name (convenience).
 */
export function release(
  modIndex: number,
  modulator: ModulatorShort,
  amount: number
) {
  releaseModulator(modIndex, ModulatorMap[modulator], amount)
}

/**
 * Decay all modulators toward their baselines.
 * Called each simulation step.
 * ALL GPU operations.
 */
export function decayModulators(modIndex: number) {
  // Dopamine decays toward 0 (baseline)
  dopamine[modIndex] = mx.add(
    mx.multiply(dopamine[modIndex], dopamineDecay[modIndex]),
    mx.multiply(
      dopamineBaseline[modIndex],
      mx.subtract(mx.array(1), dopamineDecay[modIndex])
    )
  )

  // Serotonin decays toward baseline
  serotonin[modIndex] = mx.add(
    mx.multiply(serotonin[modIndex], serotoninDecay[modIndex]),
    mx.multiply(
      serotoninBaseline[modIndex],
      mx.subtract(mx.array(1), serotoninDecay[modIndex])
    )
  )

  // Norepinephrine decays toward baseline
  norepinephrine[modIndex] = mx.add(
    mx.multiply(norepinephrine[modIndex], norepinephrineDecay[modIndex]),
    mx.multiply(
      norepinephrineBaseline[modIndex],
      mx.subtract(mx.array(1), norepinephrineDecay[modIndex])
    )
  )

  // Acetylcholine decays toward baseline
  acetylcholine[modIndex] = mx.add(
    mx.multiply(acetylcholine[modIndex], acetylcholineDecay[modIndex]),
    mx.multiply(
      acetylcholineBaseline[modIndex],
      mx.subtract(mx.array(1), acetylcholineDecay[modIndex])
    )
  )
}

/**
 * Get the combined plasticity gate.
 * This is the "third factor" that modulates STDP.
 *
 * Plasticity is enhanced when:
 * - Dopamine is high (reward signal)
 * - Norepinephrine is high (attention/arousal)
 * - Acetylcholine is high (learning mode)
 *
 * Plasticity is reduced when:
 * - Serotonin is high (satiation - no need to learn)
 *
 * Returns GPU scalar in range [0, ~4] - multiply with eligibility trace.
 * ALL GPU operations.
 */
export function getPlasticityGate(
  modIndex: number
): ReturnType<typeof mx.array> {
  // Dopamine: direct contribution (can be negative for punishment)
  // Range: -1 to 1, so add 1 to make it 0 to 2
  const daContrib = mx.add(dopamine[modIndex], mx.array(1))

  // Norepinephrine: enhances learning when aroused
  // Range: 0 to 1, so use directly
  const neContrib = mx.add(mx.array(0.5), norepinephrine[modIndex])

  // Acetylcholine: enables learning mode
  // Range: 0 to 1, so use directly
  const achContrib = mx.add(mx.array(0.5), acetylcholine[modIndex])

  // Serotonin: REDUCES learning when satiated
  // Range: 0 to 1, so invert: (1 - 5HT)
  const serContrib = mx.subtract(mx.array(1.5), serotonin[modIndex])

  // Multiply all contributions
  // Result range: roughly 0 to 4
  return mx.multiply(
    mx.multiply(daContrib, neContrib),
    mx.multiply(achContrib, serContrib)
  )
}

/**
 * Get reward signal (dopamine only).
 * This is the simple case for basic reward learning.
 * ALL GPU operations.
 */
export function getRewardSignal(modIndex: number): ReturnType<typeof mx.array> {
  return dopamine[modIndex]
}

/**
 * Set dopamine directly (for external reward signals).
 */
export function setDopamine(modIndex: number, value: number) {
  const defaults = ModulatorDefaults.dopamine
  dopamine[modIndex] = mx.clip(
    mx.array(value, mx.float32),
    mx.array(defaults.min),
    mx.array(defaults.max)
  )
}

/**
 * Set all modulators to their baselines.
 */
export function resetModulators(modIndex: number) {
  dopamine[modIndex] = mx.array(dopamineBaseline[modIndex])
  serotonin[modIndex] = mx.array(serotoninBaseline[modIndex])
  norepinephrine[modIndex] = mx.array(norepinephrineBaseline[modIndex])
  acetylcholine[modIndex] = mx.array(acetylcholineBaseline[modIndex])
}

// ============================================================================
// DERIVED VALUES - Double function pattern
// ALL values stay on GPU!
// ============================================================================

/**
 * Get derived values for a modulation system.
 * ALL values are GPU scalars.
 */
export function getModulationDerived(modIndex: number) {
  // Current levels
  const da = $derived(dopamine[modIndex])
  const ser = $derived(serotonin[modIndex])
  const ne = $derived(norepinephrine[modIndex])
  const ach = $derived(acetylcholine[modIndex])

  // Combined plasticity gate
  const plasticityGate = $derived(getPlasticityGate(modIndex))

  return () => ({
    dopamine: da,
    serotonin: ser,
    norepinephrine: ne,
    acetylcholine: ach,
    plasticityGate,
  })
}

// ============================================================================
// CONVENIENCE: Common modulation patterns
// ============================================================================

/**
 * Signal positive reward (good outcome).
 * Increases dopamine, slightly increases serotonin.
 */
export function signalReward(modIndex: number, magnitude: number = 1) {
  releaseModulator(modIndex, 'dopamine', magnitude)
  releaseModulator(modIndex, 'serotonin', magnitude * 0.2)
}

/**
 * Signal punishment/negative outcome.
 * Decreases dopamine (prediction error).
 */
export function signalPunishment(modIndex: number, magnitude: number = 1) {
  releaseModulator(modIndex, 'dopamine', -magnitude)
}

/**
 * Signal novelty/surprise.
 * Increases norepinephrine (attention) and acetylcholine (learning).
 */
export function signalNovelty(modIndex: number, magnitude: number = 1) {
  releaseModulator(modIndex, 'norepinephrine', magnitude)
  releaseModulator(modIndex, 'acetylcholine', magnitude * 0.5)
}

/**
 * Signal satiation (full, satisfied).
 * Increases serotonin, reduces norepinephrine.
 */
export function signalSatiation(modIndex: number, magnitude: number = 1) {
  releaseModulator(modIndex, 'serotonin', magnitude)
  releaseModulator(modIndex, 'norepinephrine', -magnitude * 0.3)
}

/**
 * Signal hunger/need (internal drive).
 * Decreases serotonin, increases norepinephrine.
 */
export function signalHunger(modIndex: number, magnitude: number = 1) {
  releaseModulator(modIndex, 'serotonin', -magnitude * 0.3)
  releaseModulator(modIndex, 'norepinephrine', magnitude * 0.5)
}

/**
 * Signal danger/threat.
 * Strong norepinephrine response (fight or flight).
 */
export function signalDanger(modIndex: number, magnitude: number = 1) {
  releaseModulator(modIndex, 'norepinephrine', magnitude)
  releaseModulator(modIndex, 'dopamine', -magnitude * 0.2) // Mild negative
}

// ============================================================================
// STRING ID CONVENIENCE FUNCTIONS
// These allow using network ID strings instead of indices
// ============================================================================

/**
 * Check if modulation is allocated for a network.
 */
export function isModulationAllocated(id: string): boolean {
  return registry.idToIndex.has(id)
}

/**
 * Get modulator levels for a network by ID.
 * Returns object with GPU arrays for each modulator.
 */
export function getModulatorLevels(id: string): {
  dopamine: ReturnType<typeof mx.array>
  serotonin: ReturnType<typeof mx.array>
  norepinephrine: ReturnType<typeof mx.array>
  acetylcholine: ReturnType<typeof mx.array>
} {
  const index = registry.idToIndex.get(id)
  if (index === undefined) {
    return {
      dopamine: mx.array(0, mx.float32),
      serotonin: mx.array(0.5, mx.float32),
      norepinephrine: mx.array(0.3, mx.float32),
      acetylcholine: mx.array(0.5, mx.float32),
    }
  }
  return {
    dopamine: dopamine[index],
    serotonin: serotonin[index],
    norepinephrine: norepinephrine[index],
    acetylcholine: acetylcholine[index],
  }
}

/**
 * Decay modulators by network ID.
 */
export function decayModulatorsById(id: string) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    decayModulators(index)
  }
}

/**
 * Get plasticity gate by network ID.
 */
export function getPlasticityGateById(id: string): ReturnType<typeof mx.array> {
  const index = registry.idToIndex.get(id)
  if (index === undefined) {
    return mx.array(1, mx.float32) // Neutral gate
  }
  return getPlasticityGate(index)
}

// Aliases for string-based API
export { decayModulatorsById as decayModulatorsByName }
export { getPlasticityGateById as getPlasticityGateByName }

// Re-export state for viz module (aliases)
export const dopamineLevel = dopamine
export const serotoninLevel = serotonin
export const norepinephrineLevel = norepinephrine
export const acetylcholineLevel = acetylcholine

// ============================================================================
// STRING ID VERSIONS OF SIGNAL FUNCTIONS
// ============================================================================

/**
 * Signal positive reward by network ID.
 */
export function signalRewardById(id: string, magnitude: number = 1) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    signalReward(index, magnitude)
  }
}

/**
 * Signal punishment by network ID.
 */
export function signalPunishmentById(id: string, magnitude: number = 1) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    signalPunishment(index, magnitude)
  }
}

/**
 * Signal novelty by network ID.
 */
export function signalNoveltyById(id: string, magnitude: number = 1) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    signalNovelty(index, magnitude)
  }
}

/**
 * Signal satiation by network ID.
 */
export function signalSatiationById(id: string, magnitude: number = 1) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    signalSatiation(index, magnitude)
  }
}

/**
 * Signal hunger by network ID.
 */
export function signalHungerById(id: string, magnitude: number = 1) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    signalHunger(index, magnitude)
  }
}

/**
 * Signal danger by network ID.
 */
export function signalDangerById(id: string, magnitude: number = 1) {
  const index = registry.idToIndex.get(id)
  if (index !== undefined) {
    signalDanger(index, magnitude)
  }
}
