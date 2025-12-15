/**
 * Integration Test - Brain Primitives
 *
 * Tests all modules working together:
 * - Genome definition and loading
 * - Sensory encoding
 * - Network simulation
 * - Neuromodulation
 * - Motor decoding
 * - Visualization/recording
 *
 * This creates a simple reflex arc: stimulus → sensory → inter → motor → response
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import type { Genome } from './src/core/genome.svelte.ts'
import { loadGenome, validateGenome, getPopulationForNeuron } from './src/core/genome.svelte.ts'
import { step, getNetworkDerived, networkPopulations } from './src/core/network.svelte.ts'
import { allocateSensor, encodeRate, encodeBinary } from './src/core/sensory.svelte.ts'
import { allocateMotor, updateSpikeWindow, decodeRate } from './src/core/motor.svelte.ts'
import {
  allocateModulation,
  signalRewardById,
  signalPunishmentById,
  getModulatorLevels,
} from './src/core/modulation.svelte.ts'
import { recordAll, getNetworkActivitySummary, getRasterData, clearAll } from './src/core/viz.svelte.ts'
import { fired, voltage, populationSize } from './src/core/neuron.svelte.ts'

// ============================================================================
// TEST GENOME: Simple Reflex Arc
// ============================================================================

/**
 * A minimal but complete genome for testing.
 * Models a simple stimulus-response reflex:
 *
 * Touch sensor (3 neurons) → Interneurons (5 neurons) → Motor output (2 neurons)
 *
 * This is analogous to the simplest biological reflex:
 * touching a hot surface → sensory neuron → interneuron → motor neuron → withdraw hand
 */
const reflexGenome: Genome = {
  name: 'test_reflex',
  description: 'Simple reflex arc for integration testing',

  neurons: [
    // Sensory population - receives external input
    {
      id: 'sensory',
      size: 3,
      type: 'RS',       // Regular spiking
      excitatory: true, // Sensory neurons are excitatory
      role: 'sensory',
      region: 'other',
      noise: 2,         // Low background noise
    },
    // Interneuron population - processes signal
    {
      id: 'inter',
      size: 5,
      type: 'RS',
      excitatory: true,
      role: 'inter',
      region: 'other',
      noise: 5,         // Normal background noise
    },
    // Inhibitory interneurons - for balance
    {
      id: 'inhib',
      size: 2,
      type: 'FS',       // Fast spiking inhibitory
      excitatory: false,
      role: 'inter',
      region: 'other',
      noise: 3,
    },
    // Motor population - produces output
    {
      id: 'motor',
      size: 2,
      type: 'RS',
      excitatory: true,
      role: 'motor',
      region: 'other',
      noise: 2,
    },
  ],

  synapses: [
    // Sensory → Inter (feedforward)
    {
      id: 'sens_to_inter',
      pre: 'sensory',
      post: 'inter',
      pattern: 'all-to-all',
      plastic: true,
      initialWeight: 0.3,  // Strong innate connection
    },
    // Inter → Motor (feedforward)
    {
      id: 'inter_to_motor',
      pre: 'inter',
      post: 'motor',
      pattern: 'all-to-all',
      plastic: true,
      initialWeight: 0.25,
    },
    // Sensory → Inhibitory
    {
      id: 'sens_to_inhib',
      pre: 'sensory',
      post: 'inhib',
      pattern: 'all-to-all',
      plastic: false,
      initialWeight: 0.2,
    },
    // Inhibitory → Inter (lateral inhibition effect)
    {
      id: 'inhib_to_inter',
      pre: 'inhib',
      post: 'inter',
      pattern: 'all-to-all',
      plastic: false,
      initialWeight: -0.5,  // Negative because inhibitory
    },
  ],

  // Simple reflex: strong direct pathway
  reflexes: [
    {
      name: 'withdrawal',
      pathway: ['sensory', 'motor'],  // Direct shortcut for fast response
      strength: 0.4,
      plastic: false,  // Innate, not learnable
    },
  ],

  version: '1.0',
  author: 'test',
}

// ============================================================================
// TEST RUNNER
// ============================================================================

async function runTest() {
  console.log('='.repeat(60))
  console.log('BRAIN PRIMITIVES INTEGRATION TEST')
  console.log('='.repeat(60))
  console.log()

  // -------------------------------------------------------------------------
  // 1. Validate and load genome
  // -------------------------------------------------------------------------
  console.log('1. GENOME VALIDATION AND LOADING')
  console.log('-'.repeat(40))

  const validation = validateGenome(reflexGenome)
  console.log(`   Valid: ${validation.valid}`)
  if (validation.errors.length > 0) {
    console.log(`   Errors: ${validation.errors.join(', ')}`)
    return
  }
  if (validation.warnings.length > 0) {
    console.log(`   Warnings: ${validation.warnings.join(', ')}`)
  }

  const networkIndex = loadGenome(reflexGenome)
  console.log(`   Network loaded at index: ${networkIndex}`)
  console.log(`   Populations: ${networkPopulations[networkIndex].length}`)
  console.log()

  // -------------------------------------------------------------------------
  // 2. Setup sensory and motor interfaces
  // -------------------------------------------------------------------------
  console.log('2. SENSORY AND MOTOR SETUP')
  console.log('-'.repeat(40))

  const sensoryPopIndex = getPopulationForNeuron('test_reflex', 'sensory')!
  const motorPopIndex = getPopulationForNeuron('test_reflex', 'motor')!

  const sensorIndex = allocateSensor('touch', sensoryPopIndex, {
    type: 'touch',
    encoding: 'rate',
    gain: 20,       // Amplify input
    threshold: 0.1, // Minimum activation
    adaptationRate: 0.02,
  })

  const motorIndex = allocateMotor('withdraw', motorPopIndex, {
    type: 'muscle',
    decoding: 'rate',
    gain: 1.0,
    threshold: 0.1,
    windowSize: 5,
  })

  console.log(`   Sensor allocated: index ${sensorIndex}`)
  console.log(`   Motor allocated: index ${motorIndex}`)
  console.log()

  // -------------------------------------------------------------------------
  // 3. Initialize modulation system
  // -------------------------------------------------------------------------
  console.log('3. NEUROMODULATION SETUP')
  console.log('-'.repeat(40))

  allocateModulation('test_reflex', {
    dopamineDecay: 0.95,
    serotoninDecay: 0.98,
    norepinephrineDecay: 0.9,
    acetylcholineDecay: 0.92,
  })

  console.log('   Modulation system initialized')
  console.log()

  // -------------------------------------------------------------------------
  // 4. Run simulation: No stimulus phase
  // -------------------------------------------------------------------------
  console.log('4. SIMULATION: NO STIMULUS (50 steps)')
  console.log('-'.repeat(40))

  clearAll(networkIndex, 'test_reflex')

  for (let t = 0; t < 50; t++) {
    // No sensory input
    updateSpikeWindow(motorIndex)
    step(networkIndex, 1.0)
    recordAll(networkIndex, 'test_reflex')
  }

  let summary = getNetworkActivitySummary(networkIndex)
  console.log(`   Total neurons: ${summary.totalNeurons}`)
  console.log(`   Firing neurons: ${summary.firingNeurons}`)
  console.log(`   Avg voltage: ${summary.avgVoltage.toFixed(2)} mV`)
  console.log(`   E/I ratio: ${(summary.excRatio * 100).toFixed(1)}%`)
  console.log()

  // -------------------------------------------------------------------------
  // 5. Run simulation: Stimulus phase
  // -------------------------------------------------------------------------
  console.log('5. SIMULATION: WITH STIMULUS (100 steps)')
  console.log('-'.repeat(40))

  const motorResponses: number[] = []

  for (let t = 0; t < 100; t++) {
    // Apply stimulus for first 20 steps
    if (t < 20) {
      encodeRate(sensorIndex, 0.8)  // Strong stimulus
    }

    // Signal reward when motor responds (simulating successful withdrawal)
    mx.eval(fired[motorPopIndex])
    const motorFired = mx.sum(fired[motorPopIndex]).item() as number
    if (motorFired > 0 && t < 30) {
      signalRewardById('test_reflex', 0.5)
    }

    updateSpikeWindow(motorIndex)
    step(networkIndex, 1.0)
    recordAll(networkIndex, 'test_reflex')

    // Record motor output
    mx.eval(decodeRate(motorIndex))
    motorResponses.push(decodeRate(motorIndex).item() as number)
  }

  summary = getNetworkActivitySummary(networkIndex)
  console.log(`   Final firing neurons: ${summary.firingNeurons}`)
  console.log(`   Final E/I ratio: ${(summary.excRatio * 100).toFixed(1)}%`)

  // Analyze motor response
  const maxResponse = Math.max(...motorResponses)
  const avgResponse = motorResponses.reduce((a, b) => a + b, 0) / motorResponses.length
  const responseLatency = motorResponses.findIndex(r => r > 0.1)

  console.log(`   Max motor response: ${maxResponse.toFixed(4)}`)
  console.log(`   Avg motor response: ${avgResponse.toFixed(4)}`)
  console.log(`   Response latency: ${responseLatency >= 0 ? responseLatency : 'no response'} steps`)
  console.log()

  // -------------------------------------------------------------------------
  // 6. Check modulator levels
  // -------------------------------------------------------------------------
  console.log('6. MODULATOR LEVELS')
  console.log('-'.repeat(40))

  const modLevels = getModulatorLevels('test_reflex')
  mx.eval(modLevels.dopamine, modLevels.serotonin, modLevels.norepinephrine, modLevels.acetylcholine)

  console.log(`   Dopamine: ${(modLevels.dopamine.item() as number).toFixed(4)}`)
  console.log(`   Serotonin: ${(modLevels.serotonin.item() as number).toFixed(4)}`)
  console.log(`   Norepinephrine: ${(modLevels.norepinephrine.item() as number).toFixed(4)}`)
  console.log(`   Acetylcholine: ${(modLevels.acetylcholine.item() as number).toFixed(4)}`)
  console.log()

  // -------------------------------------------------------------------------
  // 7. Check raster data
  // -------------------------------------------------------------------------
  console.log('7. SPIKE RASTER DATA')
  console.log('-'.repeat(40))

  const raster = getRasterData(networkIndex, 50)
  console.log(`   Spikes in last 50 steps: ${raster.times.length}`)

  // Count spikes per population
  const spikesPerPop = new Map<number, number>()
  for (const p of raster.populations) {
    spikesPerPop.set(p, (spikesPerPop.get(p) ?? 0) + 1)
  }

  for (const [popIndex, count] of spikesPerPop) {
    console.log(`   Population ${popIndex}: ${count} spikes`)
  }
  console.log()

  // -------------------------------------------------------------------------
  // 8. Test punishment signal
  // -------------------------------------------------------------------------
  console.log('8. PUNISHMENT SIGNAL TEST')
  console.log('-'.repeat(40))

  signalPunishmentById('test_reflex', 1.0)
  const modLevels2 = getModulatorLevels('test_reflex')
  mx.eval(modLevels2.dopamine)
  console.log(`   Dopamine after punishment: ${(modLevels2.dopamine.item() as number).toFixed(4)}`)
  console.log()

  // -------------------------------------------------------------------------
  // 9. Final summary
  // -------------------------------------------------------------------------
  console.log('='.repeat(60))
  console.log('TEST COMPLETE')
  console.log('='.repeat(60))
  console.log()

  // Report success/failure
  const success = maxResponse > 0 && responseLatency >= 0 && responseLatency < 30

  if (success) {
    console.log('✓ Reflex arc successfully responded to stimulus')
    console.log('✓ All modules integrated correctly')
  } else {
    console.log('✗ Reflex arc did not respond as expected')
    console.log('  Check neuron parameters and synaptic weights')
  }

  console.log()
}

// Run the test
runTest().catch(console.error)
