/**
 * Motor System - Brain to World Interface
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - ALL computation on GPU - NEVER convert to JS
 *
 * This module handles:
 * - Decoding motor neuron activity into actions
 * - Rate decoding, population decoding, winner-take-all
 * - Motor fatigue and recovery
 * - Action thresholds and smoothing
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'
import {
  fired,
  voltage,
  populationSize,
} from './neuron.svelte.ts'

// ============================================================================
// MOTOR TYPES
// ============================================================================

export type MotorType = 'muscle' | 'gland' | 'behavior' | 'locomotion' | 'custom'

export type DecodingType = 'rate' | 'population' | 'winner_take_all' | 'labeled_line'

// ============================================================================
// MOTOR REGISTRY
// ============================================================================

export const motorRegistry = $state({
  idToIndex: new SvelteMap<string, number>(),
  indexToId: new SvelteMap<number, string>(),
  allocatedIndices: new Set<number>(),
  freeIndices: [] as number[],
  nextIndex: 0,
})

// ============================================================================
// MOTOR STATE (DIRECT EXPORTS)
// ============================================================================

// Motor metadata
export const motorType = $state<MotorType[]>([])
export const motorDecoding = $state<DecodingType[]>([])
export const motorPopulationIndex = $state<number[]>([])  // Which neuron population drives this motor

// Action mapping - what each motor output means
export const actionName = $state<string[]>([])  // Human-readable action name
export const actionDimensions = $state<number[]>([])  // How many output dimensions

// Decoding parameters (GPU arrays)
export const motorGain = $state<ReturnType<typeof mx.array>[]>([])       // Scaling factor
export const motorBias = $state<ReturnType<typeof mx.array>[]>([])       // Output offset
export const motorThreshold = $state<ReturnType<typeof mx.array>[]>([])  // Minimum activity for action

// Output smoothing (for continuous actions)
export const outputSmoothing = $state<ReturnType<typeof mx.array>[]>([])  // Exponential smoothing factor
export const lastOutput = $state<ReturnType<typeof mx.array>[]>([])       // Previous output for smoothing

// Motor fatigue (GPU arrays)
export const fatigueLevel = $state<ReturnType<typeof mx.array>[]>([])     // Current fatigue (0 = none, 1 = exhausted)
export const fatigueRate = $state<ReturnType<typeof mx.array>[]>([])      // How fast muscles tire
export const fatigueRecovery = $state<ReturnType<typeof mx.array>[]>([])  // How fast they recover

// Spike count window for rate decoding
export const spikeWindow = $state<ReturnType<typeof mx.array>[][]>([])  // Rolling window of spike counts
export const windowSize = $state<number[]>([])  // How many timesteps to average

// ============================================================================
// MOTOR ALLOCATION
// ============================================================================

export interface MotorOptions {
  type?: MotorType
  decoding?: DecodingType
  actionName?: string
  gain?: number
  bias?: number
  threshold?: number
  smoothing?: number
  fatigueRate?: number
  fatigueRecovery?: number
  windowSize?: number
}

/**
 * Allocate a motor interface for a population.
 * Returns the motor index.
 *
 * @param id - Unique identifier for this motor
 * @param populationIndex - The neuron population that drives this motor
 * @param options - Configuration options
 */
export function allocateMotor(
  id: string,
  populationIndex: number,
  options: MotorOptions = {}
): number {
  // Check if already exists
  const existing = motorRegistry.idToIndex.get(id)
  if (existing !== undefined) return existing

  // Allocate index
  let index: number
  if (motorRegistry.freeIndices.length > 0) {
    index = motorRegistry.freeIndices.pop()!
  } else {
    index = motorRegistry.nextIndex++
  }

  // Update registry
  motorRegistry.idToIndex.set(id, index)
  motorRegistry.indexToId.set(index, id)
  motorRegistry.allocatedIndices.add(index)

  const size = populationSize[populationIndex]
  const decoding = options.decoding ?? 'rate'

  // Initialize metadata
  motorType[index] = options.type ?? 'custom'
  motorDecoding[index] = decoding
  motorPopulationIndex[index] = populationIndex
  actionName[index] = options.actionName ?? id

  // Action dimensions based on decoding type
  switch (decoding) {
    case 'rate':
      actionDimensions[index] = 1  // Single scalar output
      break
    case 'population':
      actionDimensions[index] = 1  // Weighted average
      break
    case 'winner_take_all':
      actionDimensions[index] = 1  // Index of winner
      break
    case 'labeled_line':
      actionDimensions[index] = size  // One output per neuron
      break
  }

  // Initialize decoding parameters (GPU)
  motorGain[index] = mx.array(options.gain ?? 1.0, mx.float32)
  motorBias[index] = mx.array(options.bias ?? 0.0, mx.float32)
  motorThreshold[index] = mx.array(options.threshold ?? 0.0, mx.float32)

  // Initialize smoothing - default 0.8 for stability with sparse spiking
  // With smoothing=0.8: output = 0.8*lastOutput + 0.2*newOutput (slow decay)
  outputSmoothing[index] = mx.array(options.smoothing ?? 0.8, mx.float32)
  lastOutput[index] = mx.zeros([actionDimensions[index]], mx.float32)

  // Initialize fatigue
  fatigueLevel[index] = mx.zeros([size], mx.float32)
  fatigueRate[index] = mx.array(options.fatigueRate ?? 0.001, mx.float32)
  fatigueRecovery[index] = mx.array(options.fatigueRecovery ?? 0.01, mx.float32)

  // Initialize spike window for rate decoding
  // Default 20ms window to capture at least one spike from RS neurons (ISI ~15-20ms)
  const winSize = options.windowSize ?? 20
  windowSize[index] = winSize
  spikeWindow[index] = []
  for (let i = 0; i < winSize; i++) {
    spikeWindow[index].push(mx.zeros([size], mx.float32))
  }

  return index
}

/**
 * Get motor index by id.
 */
export function getMotorIndex(id: string): number | undefined {
  return motorRegistry.idToIndex.get(id)
}

/**
 * Release a motor.
 */
export function releaseMotor(id: string): void {
  const index = motorRegistry.idToIndex.get(id)
  if (index === undefined) return

  motorRegistry.idToIndex.delete(id)
  motorRegistry.indexToId.delete(index)
  motorRegistry.allocatedIndices.delete(index)
  motorRegistry.freeIndices.push(index)

  // Clear state
  motorType[index] = 'custom'
  motorDecoding[index] = 'rate'
  motorPopulationIndex[index] = -1
  actionName[index] = ''
  actionDimensions[index] = 0
  spikeWindow[index] = []
}

// ============================================================================
// SPIKE WINDOW MANAGEMENT
// ============================================================================

/**
 * Update spike window with current firing and update fatigue.
 * Call this each timestep before decoding.
 * Fatigue is updated here (not in decode) to avoid side effects on read.
 *
 * @param motorIndex - Which motor
 */
export function updateSpikeWindow(motorIndex: number) {
  const popIndex = motorPopulationIndex[motorIndex]
  const size = populationSize[popIndex]
  const currentFired = fired[popIndex]

  // Convert boolean to float (1.0 for spike, 0.0 for no spike)
  const spikeFloat = mx.where(
    currentFired,
    mx.ones([size], mx.float32),
    mx.zeros([size], mx.float32)
  )

  // Shift window and add new spikes
  const window = spikeWindow[motorIndex]
  for (let i = window.length - 1; i > 0; i--) {
    window[i] = window[i - 1]
  }
  window[0] = spikeFloat

  // Update fatigue: increases with activity, recovers without
  // Moved here from decodeRate to avoid side effects on read
  const fatigueIncrease = mx.multiply(fatigueRate[motorIndex], spikeFloat)
  const fatigueDecrease = mx.multiply(fatigueRecovery[motorIndex], fatigueLevel[motorIndex])
  fatigueLevel[motorIndex] = mx.clip(
    mx.add(mx.subtract(fatigueLevel[motorIndex], fatigueDecrease), fatigueIncrease),
    mx.array(0, mx.float32),
    mx.array(1, mx.float32)
  )
}

/**
 * Get average spike rate over window.
 */
function getWindowRate(motorIndex: number): ReturnType<typeof mx.array> {
  const window = spikeWindow[motorIndex]
  if (window.length === 0) {
    return mx.zeros([populationSize[motorPopulationIndex[motorIndex]]], mx.float32)
  }

  // Sum all spikes in window
  let total = window[0]
  for (let i = 1; i < window.length; i++) {
    total = mx.add(total, window[i])
  }

  // Divide by window size
  return mx.divide(total, mx.array(window.length, mx.float32))
}

// ============================================================================
// DECODING FUNCTIONS (ALL GPU)
// ============================================================================

/**
 * Decode motor output using rate coding.
 * Pure read function - no side effects except smoothing state.
 * Average firing rate → scalar output.
 *
 * @param motorIndex - Which motor
 * @returns GPU scalar - the action magnitude
 */
export function decodeRate(motorIndex: number): ReturnType<typeof mx.array> {
  // Get average spike rate over window
  const rate = getWindowRate(motorIndex)

  // Apply fatigue: effective = rate * (1 - fatigue)
  // NOTE: fatigue is updated in updateSpikeWindow, not here
  const effectiveRate = mx.multiply(
    rate,
    mx.subtract(mx.array(1, mx.float32), fatigueLevel[motorIndex])
  )

  // Mean activity
  const meanRate = mx.mean(effectiveRate)

  // Apply threshold
  const thresholded = mx.maximum(
    mx.subtract(meanRate, motorThreshold[motorIndex]),
    mx.array(0, mx.float32)
  )

  // Scale and bias
  const output = mx.add(
    mx.multiply(motorGain[motorIndex], thresholded),
    motorBias[motorIndex]
  )

  // Apply smoothing
  const smoothing = outputSmoothing[motorIndex].item()
  if (smoothing > 0) {
    const smoothed = mx.add(
      mx.multiply(mx.array(smoothing, mx.float32), lastOutput[motorIndex]),
      mx.multiply(mx.array(1 - smoothing, mx.float32), output)
    )
    lastOutput[motorIndex] = smoothed
    return smoothed
  }

  lastOutput[motorIndex] = mx.reshape(output, [1])
  return output
}

/**
 * Decode motor output using population coding.
 * Weighted average of neuron "preferred values".
 * Each neuron represents a specific value, output is the weighted centroid.
 *
 * @param motorIndex - Which motor
 * @param preferredValues - What value each neuron represents (GPU array, length = pop size)
 * @returns GPU scalar - the decoded value
 */
export function decodePopulation(
  motorIndex: number,
  preferredValues: ReturnType<typeof mx.array>
): ReturnType<typeof mx.array> {
  // Get spike rates
  const rate = getWindowRate(motorIndex)

  // Apply fatigue (updated in updateSpikeWindow, not here)
  const effectiveRate = mx.multiply(
    rate,
    mx.subtract(mx.array(1, mx.float32), fatigueLevel[motorIndex])
  )

  // Weighted average: sum(rate * value) / sum(rate)
  const totalActivity = mx.sum(effectiveRate)
  const weightedSum = mx.sum(mx.multiply(effectiveRate, preferredValues))

  // Avoid division by zero
  const epsilon = mx.array(1e-8, mx.float32)
  const decoded = mx.divide(weightedSum, mx.add(totalActivity, epsilon))

  // Apply gain and bias
  const output = mx.add(
    mx.multiply(motorGain[motorIndex], decoded),
    motorBias[motorIndex]
  )

  // Apply smoothing
  const smoothing = outputSmoothing[motorIndex].item()
  if (smoothing > 0) {
    const smoothed = mx.add(
      mx.multiply(mx.array(smoothing, mx.float32), lastOutput[motorIndex]),
      mx.multiply(mx.array(1 - smoothing, mx.float32), output)
    )
    lastOutput[motorIndex] = smoothed
    return smoothed
  }

  lastOutput[motorIndex] = mx.reshape(output, [1])
  return output
}

/**
 * Decode motor output using winner-take-all.
 * Returns the index of the most active neuron.
 *
 * @param motorIndex - Which motor
 * @returns GPU scalar - index of winning neuron (or -1 if below threshold)
 */
export function decodeWinnerTakeAll(motorIndex: number): ReturnType<typeof mx.array> {
  // Get spike rates
  const rate = getWindowRate(motorIndex)

  // Apply fatigue (updated in updateSpikeWindow, not here)
  const effectiveRate = mx.multiply(
    rate,
    mx.subtract(mx.array(1, mx.float32), fatigueLevel[motorIndex])
  )

  // Find max
  const maxRate = mx.max(effectiveRate)

  // Check threshold
  const threshold = motorThreshold[motorIndex]
  const aboveThreshold = mx.greater(maxRate, threshold)

  // Get argmax
  const winner = mx.argmax(effectiveRate)

  // Return winner if above threshold, -1 otherwise
  const result = mx.where(
    aboveThreshold,
    winner,
    mx.array(-1, mx.int32)
  )

  lastOutput[motorIndex] = mx.array([result.item()], mx.float32)
  return result
}

/**
 * Decode motor output using labeled line coding.
 * Each neuron directly controls one output dimension.
 *
 * @param motorIndex - Which motor
 * @returns GPU array - one value per neuron
 */
export function decodeLabeledLine(motorIndex: number): ReturnType<typeof mx.array> {
  // Get spike rates
  const rate = getWindowRate(motorIndex)

  // Apply fatigue (updated in updateSpikeWindow, not here)
  const effectiveRate = mx.multiply(
    rate,
    mx.subtract(mx.array(1, mx.float32), fatigueLevel[motorIndex])
  )

  // Apply threshold
  const thresholded = mx.maximum(
    mx.subtract(effectiveRate, motorThreshold[motorIndex]),
    mx.zeros([populationSize[motorPopulationIndex[motorIndex]]], mx.float32)
  )

  // Scale and bias
  const output = mx.add(
    mx.multiply(motorGain[motorIndex], thresholded),
    motorBias[motorIndex]
  )

  // Apply smoothing (element-wise)
  const smoothing = outputSmoothing[motorIndex].item()
  if (smoothing > 0) {
    const smoothed = mx.add(
      mx.multiply(mx.array(smoothing, mx.float32), lastOutput[motorIndex]),
      mx.multiply(mx.array(1 - smoothing, mx.float32), output)
    )
    lastOutput[motorIndex] = smoothed
    return smoothed
  }

  lastOutput[motorIndex] = output
  return output
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Reset motor fatigue.
 */
export function resetFatigue(motorIndex: number) {
  const popIndex = motorPopulationIndex[motorIndex]
  const size = populationSize[popIndex]
  fatigueLevel[motorIndex] = mx.zeros([size], mx.float32)
}

/**
 * Reset spike window (clear history).
 */
export function resetSpikeWindow(motorIndex: number) {
  const size = populationSize[motorPopulationIndex[motorIndex]]
  const window = spikeWindow[motorIndex]
  for (let i = 0; i < window.length; i++) {
    window[i] = mx.zeros([size], mx.float32)
  }
}

/**
 * Reset all motor state.
 */
export function resetMotor(motorIndex: number) {
  resetFatigue(motorIndex)
  resetSpikeWindow(motorIndex)
  lastOutput[motorIndex] = mx.zeros([actionDimensions[motorIndex]], mx.float32)
}

/**
 * Decay fatigue over time.
 * Call when motor is not active.
 */
export function decayFatigue(motorIndex: number) {
  const fatigueDecrease = mx.multiply(fatigueRecovery[motorIndex], fatigueLevel[motorIndex])
  fatigueLevel[motorIndex] = mx.maximum(
    mx.subtract(fatigueLevel[motorIndex], fatigueDecrease),
    mx.zeros([populationSize[motorPopulationIndex[motorIndex]]], mx.float32)
  )
}

// ============================================================================
// DERIVED VALUES
// ============================================================================

/**
 * Get derived values for a motor.
 */
export function getMotorDerived(motorIndex: number) {
  const avgFatigue = $derived(mx.mean(fatigueLevel[motorIndex]))
  const currentOutput = $derived(lastOutput[motorIndex])

  return () => ({
    avgFatigue,     // GPU scalar
    currentOutput,  // GPU array
  })
}
