/**
 * Visualization System - Debug and Analysis Tools
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for cached data
 * - ALL computation on GPU - NEVER convert to JS except at final output
 *
 * This module provides:
 * - Raster plot data (spike times across neurons)
 * - Weight matrix visualization
 * - Activity heatmaps
 * - Modulator level tracking
 * - Network statistics over time
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'
import {
  networkPopulations,
  networkSynapseGroups,
  timestep,
  excitatorySpikeCount,
  inhibitorySpikeCount,
  getEIRatio,
} from './network.svelte.ts'
import {
  populationSize,
  voltage,
  fired,
  isExcitatory,
  THRESHOLD,
} from './neuron.svelte.ts'
import {
  weights as groupWeights,
  preIndices as groupPreIndices,
  postIndices as groupPostIndices,
  groupPrePopIndex,
  groupPostPopIndex,
} from './synapse.svelte.ts'
import {
  isModulationAllocated,
  getModulatorLevels,
  getModulationIndex,
} from './modulation.svelte.ts'

// ============================================================================
// RASTER PLOT DATA
// ============================================================================

export interface SpikeEvent {
  timestep: number
  populationIndex: number
  neuronIndex: number  // Within population
  globalIndex: number  // Across all populations
}

// Spike history storage
export const spikeHistory = $state(new SvelteMap<number, SpikeEvent[]>())  // networkIndex → events
export const maxHistoryLength = $state(1000)  // Max timesteps to keep

/**
 * Record current spikes for a network.
 * Call this each timestep to build raster data.
 *
 * @param networkIndex - Network to record
 */
export function recordSpikes(networkIndex: number) {
  const pops = networkPopulations[networkIndex]

  // Initialize history if needed
  if (!spikeHistory.has(networkIndex)) {
    spikeHistory.set(networkIndex, [])
  }
  const history = spikeHistory.get(networkIndex)!

  // Get current timestep (need to evaluate to get JS number)
  mx.eval(timestep[networkIndex])
  const t = timestep[networkIndex].item() as number

  // Track global neuron index
  let globalOffset = 0

  for (const popIndex of pops) {
    const size = populationSize[popIndex]
    const firedArray = fired[popIndex]

    // We need to find which neurons fired - this requires evaluation
    mx.eval(firedArray)

    // Convert to JS to find spike indices
    // This is one place we MUST touch CPU - for recording discrete events
    const firedData = firedArray.tolist() as boolean[]

    for (let i = 0; i < firedData.length; i++) {
      if (firedData[i]) {
        history.push({
          timestep: t,
          populationIndex: popIndex,
          neuronIndex: i,
          globalIndex: globalOffset + i,
        })
      }
    }

    globalOffset += size
  }

  // Trim history if too long
  const cutoff = t - maxHistoryLength
  while (history.length > 0 && history[0].timestep < cutoff) {
    history.shift()
  }
}

/**
 * Get raster data for visualization.
 * Returns arrays suitable for plotting.
 *
 * @param networkIndex - Network index
 * @param lastNTimesteps - How many timesteps to include (default: all)
 */
export function getRasterData(networkIndex: number, lastNTimesteps?: number): {
  times: number[]
  neurons: number[]
  populations: number[]
} {
  const history = spikeHistory.get(networkIndex) ?? []

  let filteredHistory = history
  if (lastNTimesteps !== undefined) {
    mx.eval(timestep[networkIndex])
    const currentT = timestep[networkIndex].item() as number
    const cutoff = currentT - lastNTimesteps
    filteredHistory = history.filter(e => e.timestep >= cutoff)
  }

  return {
    times: filteredHistory.map(e => e.timestep),
    neurons: filteredHistory.map(e => e.globalIndex),
    populations: filteredHistory.map(e => e.populationIndex),
  }
}

/**
 * Clear spike history for a network.
 */
export function clearSpikeHistory(networkIndex: number) {
  spikeHistory.set(networkIndex, [])
}

// ============================================================================
// WEIGHT MATRIX VISUALIZATION
// ============================================================================

/**
 * Get weight matrix as a 2D array for visualization.
 * Returns a dense matrix (may be large!).
 *
 * @param groupIndex - Synapse group index
 * @returns Object with matrix data and dimensions
 */
export function getWeightMatrix(groupIndex: number): {
  matrix: number[][]
  preSize: number
  postSize: number
  minWeight: number
  maxWeight: number
} {
  const prePopIndex = groupPrePopIndex[groupIndex]
  const postPopIndex = groupPostPopIndex[groupIndex]
  const preSize = populationSize[prePopIndex]
  const postSize = populationSize[postPopIndex]

  // Initialize dense matrix with zeros
  const matrix: number[][] = []
  for (let i = 0; i < preSize; i++) {
    matrix.push(new Array(postSize).fill(0))
  }

  // Evaluate GPU arrays
  mx.eval(groupPreIndices[groupIndex], groupPostIndices[groupIndex], groupWeights[groupIndex])

  // Convert to JS (visualization output - must touch CPU)
  const preIndices = groupPreIndices[groupIndex].tolist() as number[]
  const postIndices = groupPostIndices[groupIndex].tolist() as number[]
  const weights = groupWeights[groupIndex].tolist() as number[]

  // Fill matrix
  let minWeight = Infinity
  let maxWeight = -Infinity

  for (let i = 0; i < weights.length; i++) {
    const pre = preIndices[i]
    const post = postIndices[i]
    const w = weights[i]

    matrix[pre][post] = w
    if (w < minWeight) minWeight = w
    if (w > maxWeight) maxWeight = w
  }

  return {
    matrix,
    preSize,
    postSize,
    minWeight: minWeight === Infinity ? 0 : minWeight,
    maxWeight: maxWeight === -Infinity ? 0 : maxWeight,
  }
}

/**
 * Get weight histogram data.
 *
 * @param groupIndex - Synapse group index
 * @param bins - Number of histogram bins (default 50)
 */
export function getWeightHistogram(groupIndex: number, bins: number = 50): {
  binEdges: number[]
  counts: number[]
} {
  // Evaluate weights
  mx.eval(groupWeights[groupIndex])

  const weights = groupWeights[groupIndex].tolist() as number[]

  if (weights.length === 0) {
    return { binEdges: [], counts: [] }
  }

  // Find min/max
  let minW = weights[0]
  let maxW = weights[0]
  for (const w of weights) {
    if (w < minW) minW = w
    if (w > maxW) maxW = w
  }

  // Create bins
  const binWidth = (maxW - minW) / bins
  const binEdges: number[] = []
  const counts: number[] = new Array(bins).fill(0)

  for (let i = 0; i <= bins; i++) {
    binEdges.push(minW + i * binWidth)
  }

  // Count weights in each bin
  for (const w of weights) {
    let binIdx = Math.floor((w - minW) / binWidth)
    if (binIdx >= bins) binIdx = bins - 1
    if (binIdx < 0) binIdx = 0
    counts[binIdx]++
  }

  return { binEdges, counts }
}

// ============================================================================
// ACTIVITY HEATMAP
// ============================================================================

/**
 * Get current voltage heatmap data for a population.
 *
 * @param popIndex - Population index
 * @returns 1D array of voltages (reshape for 2D display)
 */
export function getVoltageHeatmap(popIndex: number): number[] {
  mx.eval(voltage[popIndex])
  return voltage[popIndex].tolist() as number[]
}

/**
 * Get firing state as binary heatmap.
 *
 * @param popIndex - Population index
 * @returns 1D array of 0/1 values
 */
export function getFiringHeatmap(popIndex: number): number[] {
  mx.eval(fired[popIndex])
  const firedData = fired[popIndex].tolist() as boolean[]
  return firedData.map(f => f ? 1 : 0)
}

/**
 * Get activity summary for entire network.
 */
export function getNetworkActivitySummary(networkIndex: number): {
  totalNeurons: number
  firingNeurons: number
  avgVoltage: number
  excRatio: number
} {
  const pops = networkPopulations[networkIndex]

  let totalNeurons = 0
  let firingNeurons = 0
  let voltageSum = 0

  for (const popIndex of pops) {
    const size = populationSize[popIndex]
    totalNeurons += size

    mx.eval(fired[popIndex], voltage[popIndex])
    const firedData = fired[popIndex].tolist() as boolean[]
    const voltageData = voltage[popIndex].tolist() as number[]

    for (let i = 0; i < firedData.length; i++) {
      if (firedData[i]) firingNeurons++
      voltageSum += voltageData[i]
    }
  }

  mx.eval(getEIRatio(networkIndex))
  const excRatio = getEIRatio(networkIndex).item() as number

  return {
    totalNeurons,
    firingNeurons,
    avgVoltage: totalNeurons > 0 ? voltageSum / totalNeurons : -70,
    excRatio,
  }
}

// ============================================================================
// MODULATOR TRACKING
// ============================================================================

export interface ModulatorSnapshot {
  timestep: number
  dopamine: number
  serotonin: number
  norepinephrine: number
  acetylcholine: number
}

export const modulatorHistory = $state(new SvelteMap<string, ModulatorSnapshot[]>())

/**
 * Record current modulator levels.
 *
 * @param networkId - Network id (string)
 */
export function recordModulators(networkId: string) {
  if (!isModulationAllocated(networkId)) return

  if (!modulatorHistory.has(networkId)) {
    modulatorHistory.set(networkId, [])
  }
  const history = modulatorHistory.get(networkId)!

  // Get current timestep from network
  const networkIndex = Array.from(networkPopulations.keys()).find(i =>
    networkPopulations[i]?.length > 0
  )
  let t = 0
  if (networkIndex !== undefined) {
    mx.eval(timestep[networkIndex])
    t = timestep[networkIndex].item() as number
  }

  // Get modulator levels using proper API
  const levels = getModulatorLevels(networkId)
  mx.eval(levels.dopamine, levels.serotonin, levels.norepinephrine, levels.acetylcholine)

  history.push({
    timestep: t,
    dopamine: levels.dopamine.item() as number,
    serotonin: levels.serotonin.item() as number,
    norepinephrine: levels.norepinephrine.item() as number,
    acetylcholine: levels.acetylcholine.item() as number,
  })

  // Trim history
  while (history.length > maxHistoryLength) {
    history.shift()
  }
}

/**
 * Get modulator history for plotting.
 */
export function getModulatorHistory(networkId: string): ModulatorSnapshot[] {
  return modulatorHistory.get(networkId) ?? []
}

/**
 * Clear modulator history.
 */
export function clearModulatorHistory(networkId: string) {
  modulatorHistory.set(networkId, [])
}

// ============================================================================
// TIME SERIES DATA
// ============================================================================

export interface NetworkStats {
  timestep: number
  totalSpikes: number
  excSpikes: number
  inhSpikes: number
  eiRatio: number
  avgVoltage: number
}

export const networkStatsHistory = $state(new SvelteMap<number, NetworkStats[]>())

/**
 * Record network statistics.
 *
 * @param networkIndex - Network index
 */
export function recordNetworkStats(networkIndex: number) {
  if (!networkStatsHistory.has(networkIndex)) {
    networkStatsHistory.set(networkIndex, [])
  }
  const history = networkStatsHistory.get(networkIndex)!

  // Evaluate needed values
  mx.eval(
    timestep[networkIndex],
    excitatorySpikeCount[networkIndex],
    inhibitorySpikeCount[networkIndex]
  )

  const t = timestep[networkIndex].item() as number
  const excSpikes = excitatorySpikeCount[networkIndex].item() as number
  const inhSpikes = inhibitorySpikeCount[networkIndex].item() as number

  // Get average voltage
  const pops = networkPopulations[networkIndex]
  let voltageSum = 0
  let totalNeurons = 0
  for (const popIndex of pops) {
    mx.eval(voltage[popIndex])
    const v = mx.mean(voltage[popIndex]).item() as number
    const size = populationSize[popIndex]
    voltageSum += v * size
    totalNeurons += size
  }

  const eiRatioVal = (excSpikes + inhSpikes) > 0
    ? excSpikes / (excSpikes + inhSpikes)
    : 0.8

  history.push({
    timestep: t,
    totalSpikes: excSpikes + inhSpikes,
    excSpikes,
    inhSpikes,
    eiRatio: eiRatioVal,
    avgVoltage: totalNeurons > 0 ? voltageSum / totalNeurons : -70,
  })

  // Trim history
  while (history.length > maxHistoryLength) {
    history.shift()
  }
}

/**
 * Get network stats history for plotting.
 */
export function getNetworkStatsHistory(networkIndex: number): NetworkStats[] {
  return networkStatsHistory.get(networkIndex) ?? []
}

/**
 * Clear network stats history.
 */
export function clearNetworkStatsHistory(networkIndex: number) {
  networkStatsHistory.set(networkIndex, [])
}

// ============================================================================
// COMBINED RECORDING
// ============================================================================

/**
 * Record all visualization data for a network.
 * Call this each timestep for full monitoring.
 *
 * @param networkIndex - Network index
 * @param networkId - Network id (for modulator tracking, optional)
 */
export function recordAll(networkIndex: number, networkId?: string) {
  recordSpikes(networkIndex)
  recordNetworkStats(networkIndex)
  if (networkId) {
    recordModulators(networkId)
  }
}

/**
 * Clear all visualization data for a network.
 */
export function clearAll(networkIndex: number, networkId?: string) {
  clearSpikeHistory(networkIndex)
  clearNetworkStatsHistory(networkIndex)
  if (networkId) {
    clearModulatorHistory(networkId)
  }
}

// ============================================================================
// EXPORT HELPERS
// ============================================================================

/**
 * Export raster data to CSV format string.
 */
export function exportRasterToCSV(networkIndex: number): string {
  const data = getRasterData(networkIndex)
  let csv = 'timestep,neuron,population\n'
  for (let i = 0; i < data.times.length; i++) {
    csv += `${data.times[i]},${data.neurons[i]},${data.populations[i]}\n`
  }
  return csv
}

/**
 * Export network stats to CSV format string.
 */
export function exportStatsToCSV(networkIndex: number): string {
  const history = getNetworkStatsHistory(networkIndex)
  let csv = 'timestep,totalSpikes,excSpikes,inhSpikes,eiRatio,avgVoltage\n'
  for (const s of history) {
    csv += `${s.timestep},${s.totalSpikes},${s.excSpikes},${s.inhSpikes},${s.eiRatio.toFixed(4)},${s.avgVoltage.toFixed(2)}\n`
  }
  return csv
}
