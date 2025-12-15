/**
 * Sensory System - World to Brain Interface
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - ALL computation on GPU - NEVER convert to JS
 *
 * This module handles:
 * - Encoding world stimuli into neural activity
 * - Rate coding, population coding, temporal coding
 * - Sensory adaptation (habituation to constant stimuli)
 * - Receptive field mapping
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'
import {
  injectCurrent,
  injectUniformCurrent,
  populationSize,
  voltage,
} from './neuron.svelte.ts'

// ============================================================================
// CACHED CONSTANTS (avoid creating new Metal buffers every call!)
// ============================================================================

const CONST_ZERO = mx.array(0, mx.float32)
const CONST_ONE = mx.array(1, mx.float32)

// ============================================================================
// SENSORY TYPES
// ============================================================================

export type SensorType = 'touch' | 'chemical' | 'light' | 'temperature' | 'proprioception' | 'pain' | 'custom'

export type EncodingType = 'rate' | 'population' | 'temporal' | 'place'

// ============================================================================
// SENSORY REGISTRY
// ============================================================================

export const sensoryRegistry = $state({
  idToIndex: new SvelteMap<string, number>(),
  indexToId: new SvelteMap<number, string>(),
  allocatedIndices: new Set<number>(),
  freeIndices: [] as number[],
  nextIndex: 0,
})

// ============================================================================
// SENSORY STATE (DIRECT EXPORTS)
// ============================================================================

// Sensor metadata
export const sensorType = $state<SensorType[]>([])
export const sensorEncoding = $state<EncodingType[]>([])
export const sensorPopulationIndex = $state<number[]>([])  // Which neuron population this sensor drives

// Encoding parameters (GPU arrays)
export const sensorGain = $state<ReturnType<typeof mx.array>[]>([])       // Scaling factor for input
export const sensorBaseline = $state<ReturnType<typeof mx.array>[]>([])   // Baseline current
export const sensorThreshold = $state<ReturnType<typeof mx.array>[]>([])  // Activation threshold

// Receptive fields - for spatial sensors (GPU arrays)
// Each sensor can have a receptive field that maps input space to neurons
export const receptiveFieldCenter = $state<ReturnType<typeof mx.array>[]>([])  // Center of each neuron's RF
export const receptiveFieldWidth = $state<ReturnType<typeof mx.array>[]>([])   // Width/sigma of RF

// Adaptation state (GPU arrays) - sensory fatigue
export const adaptationLevel = $state<ReturnType<typeof mx.array>[]>([])  // Current adaptation (0 = none, 1 = fully adapted)
export const adaptationRate = $state<ReturnType<typeof mx.array>[]>([])   // How fast to adapt
export const adaptationRecovery = $state<ReturnType<typeof mx.array>[]>([]) // How fast to recover

// Input history for temporal coding
export const lastInputTime = $state<ReturnType<typeof mx.array>[]>([])  // When each neuron last received input
export const inputPhase = $state<ReturnType<typeof mx.array>[]>([])     // Phase for oscillatory encoding

// ============================================================================
// SENSORY ALLOCATION
// ============================================================================

export interface SensorOptions {
  type?: SensorType
  encoding?: EncodingType
  gain?: number
  baseline?: number
  threshold?: number
  adaptationRate?: number
  adaptationRecovery?: number
  receptiveFieldWidth?: number
}

/**
 * Allocate a sensory interface for a population.
 * Returns the sensor index.
 *
 * @param id - Unique identifier for this sensor
 * @param populationIndex - The neuron population this sensor drives
 * @param options - Configuration options
 */
export function allocateSensor(
  id: string,
  populationIndex: number,
  options: SensorOptions = {}
): number {
  // Check if already exists
  const existing = sensoryRegistry.idToIndex.get(id)
  if (existing !== undefined) return existing

  // Allocate index
  let index: number
  if (sensoryRegistry.freeIndices.length > 0) {
    index = sensoryRegistry.freeIndices.pop()!
  } else {
    index = sensoryRegistry.nextIndex++
  }

  // Update registry
  sensoryRegistry.idToIndex.set(id, index)
  sensoryRegistry.indexToId.set(index, id)
  sensoryRegistry.allocatedIndices.add(index)

  const size = populationSize[populationIndex]

  // Initialize metadata
  sensorType[index] = options.type ?? 'custom'
  sensorEncoding[index] = options.encoding ?? 'rate'
  sensorPopulationIndex[index] = populationIndex

  // Initialize encoding parameters (GPU arrays)
  sensorGain[index] = mx.array(options.gain ?? 1.0, mx.float32)
  sensorBaseline[index] = mx.array(options.baseline ?? 0.0, mx.float32)
  sensorThreshold[index] = mx.array(options.threshold ?? 0.0, mx.float32)

  // Initialize receptive fields - default to no spatial tuning (all respond equally)
  // Linspace from 0 to 1 for center positions
  receptiveFieldCenter[index] = mx.divide(
    mx.arange(0, size, 1, mx.float32),
    mx.array(Math.max(size - 1, 1), mx.float32)
  )
  receptiveFieldWidth[index] = mx.full([size], options.receptiveFieldWidth ?? 0.1, mx.float32)

  // Initialize adaptation state
  adaptationLevel[index] = mx.zeros([size], mx.float32)
  adaptationRate[index] = mx.array(options.adaptationRate ?? 0.01, mx.float32)
  adaptationRecovery[index] = mx.array(options.adaptationRecovery ?? 0.001, mx.float32)

  // Initialize temporal state
  lastInputTime[index] = mx.full([size], -1000, mx.float32)  // Long ago
  inputPhase[index] = mx.zeros([size], mx.float32)

  return index
}

/**
 * Get sensor index by id.
 */
export function getSensorIndex(id: string): number | undefined {
  return sensoryRegistry.idToIndex.get(id)
}

/**
 * Release a sensor.
 */
export function releaseSensor(id: string): void {
  const index = sensoryRegistry.idToIndex.get(id)
  if (index === undefined) return

  sensoryRegistry.idToIndex.delete(id)
  sensoryRegistry.indexToId.delete(index)
  sensoryRegistry.allocatedIndices.delete(index)
  sensoryRegistry.freeIndices.push(index)

  // Clear state
  sensorType[index] = 'custom'
  sensorEncoding[index] = 'rate'
  sensorPopulationIndex[index] = -1
}

// ============================================================================
// ENCODING FUNCTIONS (ALL GPU)
// ============================================================================

/**
 * Encode a scalar stimulus using rate coding.
 * Higher stimulus → higher firing rate (more current).
 *
 * @param sensorIndex - Which sensor
 * @param stimulus - Scalar stimulus value (typically 0-1)
 * @param applyAdaptation - Whether to apply sensory adaptation
 */
export function encodeRate(
  sensorIndex: number,
  stimulus: number,
  applyAdaptation: boolean = true
) {
  const popIndex = sensorPopulationIndex[sensorIndex]
  const size = populationSize[popIndex]

  // Use mx.tidy to clean up intermediate tensors
  const result = mx.tidy(() => {
    // Convert stimulus to GPU
    const stimArray = mx.array(stimulus, mx.float32)

    // Apply threshold - use cached constants!
    const thresholded = mx.maximum(
      mx.subtract(stimArray, sensorThreshold[sensorIndex]),
      CONST_ZERO
    )

    // Scale by gain
    const scaled = mx.multiply(sensorGain[sensorIndex], thresholded)

    // Apply adaptation if enabled
    let effective = scaled
    let newAdaptation = adaptationLevel[sensorIndex]

    if (applyAdaptation) {
      // Adaptation reduces response: effective = scaled * (1 - mean_adaptation)
      const meanAdaptation = mx.mean(adaptationLevel[sensorIndex])
      effective = mx.multiply(
        scaled,
        mx.subtract(CONST_ONE, meanAdaptation)
      )

      // Update adaptation: increases with stimulus, decays without
      const adaptIncrease = mx.multiply(adaptationRate[sensorIndex], scaled)
      const adaptDecrease = mx.multiply(adaptationRecovery[sensorIndex], adaptationLevel[sensorIndex])
      newAdaptation = mx.clip(
        mx.add(mx.subtract(adaptationLevel[sensorIndex], adaptDecrease), adaptIncrease),
        CONST_ZERO,
        CONST_ONE
      )
    }

    // Add baseline and broadcast to all neurons
    const currentValue = mx.add(sensorBaseline[sensorIndex], effective)

    mx.eval(currentValue, newAdaptation)
    return { currentValue, newAdaptation, didAdapt: applyAdaptation }
  })

  // Update adaptation if enabled
  if (result.didAdapt) {
    adaptationLevel[sensorIndex] = result.newAdaptation
  }

  // Broadcast to all neurons - use uniform injection (no array allocation!)
  injectUniformCurrent(popIndex, result.currentValue.item())
}

/**
 * Encode a scalar stimulus using population coding.
 * Each neuron has a preferred value (receptive field center).
 * Neurons respond based on distance from their preferred value.
 *
 * @param sensorIndex - Which sensor
 * @param stimulus - Scalar stimulus value (typically 0-1)
 */
export function encodePopulation(
  sensorIndex: number,
  stimulus: number
) {
  const popIndex = sensorPopulationIndex[sensorIndex]
  const size = populationSize[popIndex]

  // Use mx.tidy to clean up intermediate tensors
  const result = mx.tidy(() => {
    // Convert stimulus to GPU
    const stimArray = mx.array(stimulus, mx.float32)

    // Calculate distance from each neuron's preferred value
    const centers = receptiveFieldCenter[sensorIndex]
    const widths = receptiveFieldWidth[sensorIndex]

    // Gaussian tuning curve: response = gain * exp(-((stim - center) / width)^2)
    const distance = mx.subtract(stimArray, centers)
    const normalizedDist = mx.divide(distance, widths)
    const response = mx.multiply(
      sensorGain[sensorIndex],
      mx.exp(mx.negative(mx.square(normalizedDist)))
    )

    // Apply adaptation - use cached constants!
    const adapted = mx.multiply(
      response,
      mx.subtract(CONST_ONE, adaptationLevel[sensorIndex])
    )

    // Update adaptation (per neuron based on their response)
    const adaptIncrease = mx.multiply(adaptationRate[sensorIndex], response)
    const adaptDecrease = mx.multiply(adaptationRecovery[sensorIndex], adaptationLevel[sensorIndex])
    const newAdaptation = mx.clip(
      mx.add(mx.subtract(adaptationLevel[sensorIndex], adaptDecrease), adaptIncrease),
      CONST_ZERO,
      CONST_ONE
    )

    // Add baseline
    const current = mx.add(sensorBaseline[sensorIndex], adapted)

    // Indices for injection
    const indices = mx.arange(0, size, 1, mx.int32)

    mx.eval(current, newAdaptation, indices)
    return { current, newAdaptation, indices }
  })

  // Update state and inject (outside tidy)
  adaptationLevel[sensorIndex] = result.newAdaptation
  injectCurrent(popIndex, result.indices, result.current)
}

/**
 * Encode a stimulus vector using place coding.
 * Each element of the stimulus maps to a specific neuron.
 * Useful for 1D spatial sensors (like a line of touch sensors).
 *
 * @param sensorIndex - Which sensor
 * @param stimulus - Vector of stimulus values (length must match population size)
 */
export function encodePlace(
  sensorIndex: number,
  stimulus: ReturnType<typeof mx.array>
) {
  const popIndex = sensorPopulationIndex[sensorIndex]
  const size = populationSize[popIndex]

  // Ensure stimulus is correct size
  mx.eval(stimulus)
  if (stimulus.shape[0] !== size) {
    throw new Error(`Stimulus size ${stimulus.shape[0]} doesn't match population size ${size}`)
  }

  // Use mx.tidy to clean up intermediate tensors
  const result = mx.tidy(() => {
    // Apply threshold
    const thresholded = mx.maximum(
      mx.subtract(stimulus, sensorThreshold[sensorIndex]),
      mx.zeros([size], mx.float32)
    )

    // Scale by gain
    const scaled = mx.multiply(sensorGain[sensorIndex], thresholded)

    // Apply adaptation - use cached constants!
    const adapted = mx.multiply(
      scaled,
      mx.subtract(CONST_ONE, adaptationLevel[sensorIndex])
    )

    // Update adaptation
    const adaptIncrease = mx.multiply(adaptationRate[sensorIndex], scaled)
    const adaptDecrease = mx.multiply(adaptationRecovery[sensorIndex], adaptationLevel[sensorIndex])
    const newAdaptation = mx.clip(
      mx.add(mx.subtract(adaptationLevel[sensorIndex], adaptDecrease), adaptIncrease),
      CONST_ZERO,
      CONST_ONE
    )

    // Add baseline
    const current = mx.add(sensorBaseline[sensorIndex], adapted)

    // Indices for injection
    const indices = mx.arange(0, size, 1, mx.int32)

    mx.eval(current, newAdaptation, indices)
    return { current, newAdaptation, indices }
  })

  // Update state and inject (outside tidy)
  adaptationLevel[sensorIndex] = result.newAdaptation
  injectCurrent(popIndex, result.indices, result.current)
}

/**
 * Encode a binary on/off stimulus.
 * All neurons receive current when on, none when off.
 *
 * @param sensorIndex - Which sensor
 * @param on - Whether the stimulus is active
 * @param magnitude - How much current when on
 */
export function encodeBinary(
  sensorIndex: number,
  on: boolean,
  magnitude: number = 20
) {
  if (!on) return  // No current when off

  const popIndex = sensorPopulationIndex[sensorIndex]
  const size = populationSize[popIndex]

  // Use mx.tidy to clean up intermediate tensors
  const result = mx.tidy(() => {
    // Get mean adaptation (need to eval before .item())
    const meanAdapt = mx.mean(adaptationLevel[sensorIndex])
    mx.eval(meanAdapt)
    const adaptedMag = magnitude * (1 - meanAdapt.item())

    // Update adaptation - use cached constants!
    const magArray = mx.array(magnitude, mx.float32)
    const adaptIncrease = mx.multiply(adaptationRate[sensorIndex], magArray)
    const adaptDecrease = mx.multiply(adaptationRecovery[sensorIndex], adaptationLevel[sensorIndex])
    const newAdaptation = mx.clip(
      mx.add(mx.subtract(adaptationLevel[sensorIndex], adaptDecrease), adaptIncrease),
      CONST_ZERO,
      CONST_ONE
    )

    const current = mx.full([size], adaptedMag, mx.float32)
    const indices = mx.arange(0, size, 1, mx.int32)

    mx.eval(current, newAdaptation, indices)
    return { current, newAdaptation, indices }
  })

  // Update state and inject (outside tidy)
  adaptationLevel[sensorIndex] = result.newAdaptation
  injectCurrent(popIndex, result.indices, result.current)
}

// ============================================================================
// GENERIC SEND INPUT (dispatches to specific encoder)
// ============================================================================

/**
 * Send input to a sensor based on its configured encoding type.
 * This is the main entry point for providing sensory input.
 *
 * @param sensorIndex - Which sensor to stimulate
 * @param stimulus - Scalar stimulus value (typically 0-1)
 */
export function sendInput(sensorIndex: number, stimulus: number): void {
  const encoding = sensorEncoding[sensorIndex]

  switch (encoding) {
    case 'rate':
      encodeRate(sensorIndex, stimulus)
      break
    case 'population':
      encodePopulation(sensorIndex, stimulus)
      break
    case 'place':
      encodePlace(sensorIndex, stimulus)
      break
    case 'temporal':
      // Temporal encoding uses binary with timing
      encodeBinary(sensorIndex, stimulus > 0.5)
      break
    default:
      encodeRate(sensorIndex, stimulus)
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Reset adaptation for a sensor (after stimulus removal).
 */
export function resetAdaptation(sensorIndex: number) {
  const popIndex = sensorPopulationIndex[sensorIndex]
  const size = populationSize[popIndex]
  adaptationLevel[sensorIndex] = mx.zeros([size], mx.float32)
}

/**
 * Set custom receptive field centers.
 * Useful for non-uniform spatial coverage.
 *
 * @param sensorIndex - Which sensor
 * @param centers - Array of preferred values for each neuron (0-1)
 */
export function setReceptiveFieldCenters(
  sensorIndex: number,
  centers: ReturnType<typeof mx.array>
) {
  receptiveFieldCenter[sensorIndex] = centers
}

/**
 * Set receptive field widths (tuning curve sharpness).
 *
 * @param sensorIndex - Which sensor
 * @param widths - Width/sigma for each neuron's tuning curve
 */
export function setReceptiveFieldWidths(
  sensorIndex: number,
  widths: ReturnType<typeof mx.array>
) {
  receptiveFieldWidth[sensorIndex] = widths
}

/**
 * Decay adaptation over time (call when no stimulus present).
 *
 * @param sensorIndex - Which sensor
 */
export function decayAdaptation(sensorIndex: number) {
  const adaptDecrease = mx.multiply(adaptationRecovery[sensorIndex], adaptationLevel[sensorIndex])
  adaptationLevel[sensorIndex] = mx.maximum(
    mx.subtract(adaptationLevel[sensorIndex], adaptDecrease),
    mx.zeros([populationSize[sensorPopulationIndex[sensorIndex]]], mx.float32)
  )
}

// ============================================================================
// DERIVED VALUES
// ============================================================================

/**
 * Get derived values for a sensor.
 */
export function getSensorDerived(sensorIndex: number) {
  const avgAdaptation = $derived(mx.mean(adaptationLevel[sensorIndex]))

  return () => ({
    avgAdaptation,  // GPU scalar
  })
}
