/**
 * Test 02: Synapse Transmission
 *
 * Tests: Does activity propagate through synapses?
 *
 * What we test:
 * 1. Pre-synaptic spikes cause post-synaptic current
 * 2. Weight affects transmission strength
 * 3. Excitatory synapses add positive current
 * 4. Inhibitory synapses add negative current (Dale's Law)
 * 5. No transmission without pre-synaptic spikes
 *
 * The Slap: Transmission must be CAUSAL - spikes cause current.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import {
  startSuite,
  endSuite,
  test,
  assertEqual,
  assertTrue,
  assertGreater,
  assertLess,
  assertInRange,
  getScalar,
  getArray,
  countTrue,
  withSlap,
} from './utils.ts'
import {
  allocatePopulation,
  releasePopulation,
  integrate,
  injectUniformCurrent,
  voltage,
  fired,
  current,
  isExcitatory,
  setNoiseAmplitude,
} from '../core/neuron.svelte.ts'
import {
  allocateSynapseGroup,
  releaseSynapseGroup,
  transmit,
  weights,
  createAllToAllConnectivity,
} from '../core/synapse.svelte.ts'

export function runSynapseTests() {
  startSuite('02 - Synapse Transmission')

  // -------------------------------------------------------------------------
  // Test 1: No transmission without pre-synaptic spikes
  // -------------------------------------------------------------------------
  test('No transmission without pre-synaptic spikes', () => {
    // Create two populations
    const preIndex = allocatePopulation('syn_pre_no', 5, { noise: 0 })
    const postIndex = allocatePopulation('syn_post_no', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    // Create synapse with strong weights
    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'syn_test_no',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { initialWeights: mx.full([25], 0.5, mx.float32) }
    )

    // Don't inject current to pre - no spikes should occur
    integrate(preIndex, 1.0, false)

    // Check post-synaptic current before transmission
    const currentBefore = getScalar(mx.sum(mx.abs(current[postIndex])))

    // Transmit
    transmit(synIndex)

    // Check post-synaptic current after
    const currentAfter = getScalar(mx.sum(mx.abs(current[postIndex])))

    releaseSynapseGroup('syn_test_no')
    releasePopulation('syn_pre_no')
    releasePopulation('syn_post_no')

    // Current should not change significantly without spikes
    return withSlap(
      assertLess(
        Math.abs(currentAfter - currentBefore),
        0.1,
        `Current change: ${currentAfter - currentBefore}`
      ),
      'No current without spikes proves causality'
    )
  })

  // -------------------------------------------------------------------------
  // Test 2: Transmission occurs with pre-synaptic spikes
  // -------------------------------------------------------------------------
  test('Pre-synaptic spikes cause post-synaptic current', () => {
    const preIndex = allocatePopulation('syn_pre_yes', 5, { noise: 0 })
    const postIndex = allocatePopulation('syn_post_yes', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'syn_test_yes',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { initialWeights: mx.full([25], 0.3, mx.float32) }
    )

    // Accumulate current received over all timesteps
    let totalCurrentReceived = 0
    let totalPreSpikes = 0

    // Inject strong current to make pre-neurons spike
    // Call transmit immediately after integrate while fired is valid
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(preIndex, 20)
      integrate(preIndex, 1.0, false)
      totalPreSpikes += countTrue(fired[preIndex])

      // Reset post current before transmit to measure just this step
      current[postIndex] = mx.zeros([5], mx.float32)
      transmit(synIndex)
      totalCurrentReceived += getScalar(mx.sum(current[postIndex]))
    }

    releaseSynapseGroup('syn_test_yes')
    releasePopulation('syn_pre_yes')
    releasePopulation('syn_post_yes')

    return withSlap(
      assertGreater(
        totalCurrentReceived,
        0,
        `Total current: ${totalCurrentReceived.toFixed(4)} (pre spikes: ${totalPreSpikes})`
      ),
      'Current only appears after spikes - proves transmission works'
    )
  })

  // -------------------------------------------------------------------------
  // Test 3: Excitatory synapses produce positive current
  // -------------------------------------------------------------------------
  test('Excitatory synapses produce positive current', () => {
    // RS neurons are excitatory
    const preIndex = allocatePopulation('syn_exc_pre', 5, {
      type: 'RS',
      noise: 0,
    })
    const postIndex = allocatePopulation('syn_exc_post', 5, {
      type: 'RS',
      noise: 0,
    })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'syn_exc_test',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices
    )

    // Check that weights are positive (excitatory)
    mx.eval(weights[synIndex])
    const w = getArray(weights[synIndex])
    const allPositive = w.every((v) => v >= 0)

    // Make pre spike and transmit immediately while fired is valid
    let totalPostCurrent = 0
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(preIndex, 20)
      integrate(preIndex, 1.0, false)
      current[postIndex] = mx.zeros([5], mx.float32)
      transmit(synIndex)
      totalPostCurrent += getScalar(mx.sum(current[postIndex]))
    }

    releaseSynapseGroup('syn_exc_test')
    releasePopulation('syn_exc_pre')
    releasePopulation('syn_exc_post')

    return withSlap(
      assertTrue(
        allPositive && totalPostCurrent >= 0,
        `Weights positive: ${allPositive}, Current: ${totalPostCurrent.toFixed(4)}`
      ),
      "Dale's Law: excitatory neurons have positive weights"
    )
  })

  // -------------------------------------------------------------------------
  // Test 4: Inhibitory synapses produce negative current
  // -------------------------------------------------------------------------
  test('Inhibitory synapses produce negative current', () => {
    // FS neurons are inhibitory
    const preIndex = allocatePopulation('syn_inh_pre', 5, {
      type: 'FS',
      noise: 0,
    })
    const postIndex = allocatePopulation('syn_inh_post', 5, {
      type: 'RS',
      noise: 0,
    })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'syn_inh_test',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices
    )

    // Check that weights are negative (inhibitory)
    mx.eval(weights[synIndex])
    const w = getArray(weights[synIndex])
    const allNegative = w.every((v) => v <= 0)

    // Make pre spike and transmit immediately while fired is valid
    let totalPostCurrent = 0
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(preIndex, 20)
      integrate(preIndex, 1.0, false)
      current[postIndex] = mx.zeros([5], mx.float32)
      transmit(synIndex)
      totalPostCurrent += getScalar(mx.sum(current[postIndex]))
    }

    releaseSynapseGroup('syn_inh_test')
    releasePopulation('syn_inh_pre')
    releasePopulation('syn_inh_post')

    return withSlap(
      assertTrue(
        allNegative && totalPostCurrent <= 0,
        `Weights negative: ${allNegative}, Current: ${totalPostCurrent.toFixed(4)}`
      ),
      "Dale's Law: inhibitory neurons have negative weights"
    )
  })

  // -------------------------------------------------------------------------
  // Test 5: Higher weights = more current
  // -------------------------------------------------------------------------
  test('Higher weights produce more current', () => {
    const preIndex = allocatePopulation('syn_weight_pre', 5, { noise: 0 })
    const postIndexLow = allocatePopulation('syn_weight_post_low', 5, {
      noise: 0,
    })
    const postIndexHigh = allocatePopulation('syn_weight_post_high', 5, {
      noise: 0,
    })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndexLow, 0)
    setNoiseAmplitude(postIndexHigh, 0)

    const conn = createAllToAllConnectivity(5, 5)

    const synLow = allocateSynapseGroup(
      'syn_low',
      preIndex,
      postIndexLow,
      mx.array(conn.preIndices),
      mx.array(conn.postIndices),
      { initialWeights: mx.full([25], 0.1, mx.float32) }
    )

    const synHigh = allocateSynapseGroup(
      'syn_high',
      preIndex,
      postIndexHigh,
      mx.array(conn.preIndices),
      mx.array(conn.postIndices),
      { initialWeights: mx.full([25], 0.5, mx.float32) }
    )

    // Make pre spike and transmit immediately while fired is valid
    let totalLowCurrent = 0
    let totalHighCurrent = 0
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(preIndex, 20)
      integrate(preIndex, 1.0, false)

      current[postIndexLow] = mx.zeros([5], mx.float32)
      current[postIndexHigh] = mx.zeros([5], mx.float32)

      transmit(synLow)
      transmit(synHigh)

      totalLowCurrent += getScalar(mx.sum(current[postIndexLow]))
      totalHighCurrent += getScalar(mx.sum(current[postIndexHigh]))
    }

    releaseSynapseGroup('syn_low')
    releaseSynapseGroup('syn_high')
    releasePopulation('syn_weight_pre')
    releasePopulation('syn_weight_post_low')
    releasePopulation('syn_weight_post_high')

    return withSlap(
      assertGreater(
        totalHighCurrent,
        totalLowCurrent,
        `High: ${totalHighCurrent.toFixed(4)} > Low: ${totalLowCurrent.toFixed(4)}`
      ),
      'Weight scaling proves transmission uses weights'
    )
  })

  // -------------------------------------------------------------------------
  // Test 6: Transmission can cause post-synaptic spikes
  // -------------------------------------------------------------------------
  test('Strong transmission can trigger post-synaptic spikes', () => {
    const preIndex = allocatePopulation('syn_chain_pre', 10, { noise: 0 })
    const postIndex = allocatePopulation('syn_chain_post', 10, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    // 10 pre × weight = current per post. Need ~10-15 to spike, so weight ~1.5
    const conn = createAllToAllConnectivity(10, 10)
    const synIndex = allocateSynapseGroup(
      'syn_chain',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { initialWeights: mx.full([100], 1.5, mx.float32) }  // 10 pre × 1.5 = 15 current
    )

    let postSpikes = 0

    // Drive pre-neurons strongly
    for (let t = 0; t < 50; t++) {
      injectUniformCurrent(preIndex, 20)
      integrate(preIndex, 1.0, false)
      transmit(synIndex)
      integrate(postIndex, 1.0, false)
      postSpikes += countTrue(fired[postIndex])
    }

    releaseSynapseGroup('syn_chain')
    releasePopulation('syn_chain_pre')
    releasePopulation('syn_chain_post')

    return withSlap(
      assertGreater(postSpikes, 0, `Post-synaptic spikes: ${postSpikes}`),
      'Chain reaction proves signal propagation through network'
    )
  })

  endSuite()
}

// Run if executed directly
if (import.meta) {
  runSynapseTests()
}
