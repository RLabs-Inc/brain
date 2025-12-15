/**
 * Test 03: STDP (Spike-Timing-Dependent Plasticity)
 *
 * Tests: Do weights change based on spike timing?
 *
 * STDP Rule:
 * - Pre before Post (causal) → strengthen (LTP)
 * - Post before Pre (acausal) → weaken (LTD)
 *
 * What we test:
 * 1. Weights change with activity (eligibility traces)
 * 2. Reward signal converts eligibility to weight change
 * 3. No weight change without reward (three-factor)
 * 4. Weights respect bounds (Dale's Law)
 *
 * The Slap: Learning must require the RIGHT conditions.
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
  withSlap,
  countTrue,
} from './utils.ts'
import {
  allocatePopulation,
  releasePopulation,
  integrate,
  injectUniformCurrent,
  setNoiseAmplitude,
  fired,
} from '../core/neuron.svelte.ts'
import {
  allocateSynapseGroup,
  releaseSynapseGroup,
  transmit,
  updateTraces,
  applySTDP,
  applyReward,
  resetLearning,
  weights,
  eligibility,
  minWeight,
  maxWeight,
  createAllToAllConnectivity,
} from '../core/synapse.svelte.ts'

export function runSTDPTests() {
  startSuite('03 - STDP Learning')

  // -------------------------------------------------------------------------
  // Test 1: Eligibility traces build with activity
  // -------------------------------------------------------------------------
  test('Eligibility traces build with correlated activity', () => {
    const preIndex = allocatePopulation('stdp_pre1', 5, { noise: 0 })
    const postIndex = allocatePopulation('stdp_post1', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'stdp_syn1',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { plastic: true }
    )

    // Initial eligibility should be zero
    mx.eval(eligibility[synIndex])
    const eligBefore = getScalar(mx.sum(mx.abs(eligibility[synIndex])))

    // Drive both populations to spike together (correlated activity)
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(preIndex, 20)
      injectUniformCurrent(postIndex, 20)
      integrate(preIndex, 1.0, false)
      integrate(postIndex, 1.0, false)
      transmit(synIndex)
      updateTraces(synIndex)
      applySTDP(synIndex, false) // Build eligibility, don't update weights
    }

    mx.eval(eligibility[synIndex])
    const eligAfter = getScalar(mx.sum(mx.abs(eligibility[synIndex])))

    releaseSynapseGroup('stdp_syn1')
    releasePopulation('stdp_pre1')
    releasePopulation('stdp_post1')

    return withSlap(
      assertGreater(
        eligAfter,
        eligBefore,
        `Eligibility: ${eligBefore.toFixed(4)} → ${eligAfter.toFixed(4)}`
      ),
      'Eligibility builds with activity - proves trace mechanism works'
    )
  })

  // -------------------------------------------------------------------------
  // Test 2: No weight change without reward (three-factor)
  // -------------------------------------------------------------------------
  test('No weight change without reward signal', () => {
    const preIndex = allocatePopulation('stdp_pre2', 5, { noise: 0 })
    const postIndex = allocatePopulation('stdp_post2', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'stdp_syn2',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { plastic: true, initialWeights: mx.full([25], 0.25, mx.float32) }
    )

    mx.eval(weights[synIndex])
    const weightsBefore = getScalar(mx.mean(weights[synIndex]))

    // Activity without reward
    for (let t = 0; t < 50; t++) {
      injectUniformCurrent(preIndex, 20)
      injectUniformCurrent(postIndex, 20)
      integrate(preIndex, 1.0, false)
      integrate(postIndex, 1.0, false)
      transmit(synIndex)
      updateTraces(synIndex)
      applySTDP(synIndex, false)
      // NO applyReward() - no reward signal
    }

    mx.eval(weights[synIndex])
    const weightsAfter = getScalar(mx.mean(weights[synIndex]))
    const change = Math.abs(weightsAfter - weightsBefore)

    releaseSynapseGroup('stdp_syn2')
    releasePopulation('stdp_pre2')
    releasePopulation('stdp_post2')

    return withSlap(
      assertLess(
        change,
        0.01,
        `Weight change without reward: ${change.toFixed(6)}`
      ),
      "Three-factor learning: eligibility alone doesn't change weights"
    )
  })

  // -------------------------------------------------------------------------
  // Test 3: Reward causes weight change
  // -------------------------------------------------------------------------
  test('Reward signal converts eligibility to weight change', () => {
    const preIndex = allocatePopulation('stdp_pre3', 5, { noise: 0 })
    const postIndex = allocatePopulation('stdp_post3', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'stdp_syn3',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { plastic: true, initialWeights: mx.full([25], 0.25, mx.float32) }
    )

    mx.eval(weights[synIndex])
    const weightsBefore = getScalar(mx.mean(weights[synIndex]))

    // Build eligibility
    for (let t = 0; t < 30; t++) {
      injectUniformCurrent(preIndex, 20)
      injectUniformCurrent(postIndex, 20)
      integrate(preIndex, 1.0, false)
      integrate(postIndex, 1.0, false)
      transmit(synIndex)
      updateTraces(synIndex)
      applySTDP(synIndex, false)
    }

    // Apply reward
    const reward = mx.array(1.0, mx.float32)
    applyReward(synIndex, reward)

    mx.eval(weights[synIndex])
    const weightsAfter = getScalar(mx.mean(weights[synIndex]))
    const change = weightsAfter - weightsBefore

    releaseSynapseGroup('stdp_syn3')
    releasePopulation('stdp_pre3')
    releasePopulation('stdp_post3')

    return withSlap(
      assertTrue(
        Math.abs(change) > 0.001,
        `Weight change with reward: ${change.toFixed(6)}`
      ),
      'Reward converts eligibility to learning - three-factor rule works'
    )
  })

  // -------------------------------------------------------------------------
  // Test 4: Weights stay within bounds
  // -------------------------------------------------------------------------
  test('Weights respect min/max bounds', () => {
    const preIndex = allocatePopulation('stdp_pre4', 5, {
      type: 'RS',
      noise: 0,
    })
    const postIndex = allocatePopulation('stdp_post4', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'stdp_syn4',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { plastic: true }
    )

    // Get bounds
    mx.eval(minWeight[synIndex], maxWeight[synIndex])
    const minW = getScalar(minWeight[synIndex])
    const maxW = getScalar(maxWeight[synIndex])

    // Heavy repeated learning
    for (let epoch = 0; epoch < 10; epoch++) {
      for (let t = 0; t < 30; t++) {
        injectUniformCurrent(preIndex, 20)
        injectUniformCurrent(postIndex, 20)
        integrate(preIndex, 1.0, false)
        integrate(postIndex, 1.0, false)
        transmit(synIndex)
        updateTraces(synIndex)
        applySTDP(synIndex, false)
      }
      applyReward(synIndex, mx.array(1.0, mx.float32))
    }

    mx.eval(weights[synIndex])
    const w = getArray(weights[synIndex])
    const allInBounds = w.every((v) => v >= minW - 0.001 && v <= maxW + 0.001)
    const actualMin = Math.min(...w)
    const actualMax = Math.max(...w)

    releaseSynapseGroup('stdp_syn4')
    releasePopulation('stdp_pre4')
    releasePopulation('stdp_post4')

    return withSlap(
      assertTrue(
        allInBounds,
        `Bounds [${minW}, ${maxW}], Actual [${actualMin.toFixed(
          3
        )}, ${actualMax.toFixed(3)}]`
      ),
      "Dale's Law enforced: weights can't exceed biological bounds"
    )
  })

  // -------------------------------------------------------------------------
  // Test 5: Non-plastic synapses don't learn
  // -------------------------------------------------------------------------
  test('Non-plastic synapses do not change', () => {
    const preIndex = allocatePopulation('stdp_pre5', 5, { noise: 0 })
    const postIndex = allocatePopulation('stdp_post5', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'stdp_syn5',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { plastic: false, initialWeights: mx.full([25], 0.3, mx.float32) } // Non-plastic!
    )

    mx.eval(weights[synIndex])
    const weightsBefore = getArray(weights[synIndex])

    // Heavy activity with reward
    for (let t = 0; t < 50; t++) {
      injectUniformCurrent(preIndex, 20)
      injectUniformCurrent(postIndex, 20)
      integrate(preIndex, 1.0, false)
      integrate(postIndex, 1.0, false)
      transmit(synIndex)
      updateTraces(synIndex)
      applySTDP(synIndex, false)
    }
    applyReward(synIndex, mx.array(1.0, mx.float32))

    mx.eval(weights[synIndex])
    const weightsAfter = getArray(weights[synIndex])

    // Check all weights unchanged
    let maxChange = 0
    for (let i = 0; i < weightsBefore.length; i++) {
      maxChange = Math.max(
        maxChange,
        Math.abs(weightsAfter[i] - weightsBefore[i])
      )
    }

    releaseSynapseGroup('stdp_syn5')
    releasePopulation('stdp_pre5')
    releasePopulation('stdp_post5')

    return withSlap(
      assertEqual(maxChange, 0, `Max weight change: ${maxChange.toFixed(8)}`),
      'Non-plastic flag respected - innate wiring preserved'
    )
  })

  // -------------------------------------------------------------------------
  // Test 6: Reset learning clears eligibility
  // -------------------------------------------------------------------------
  test('Reset learning clears eligibility traces', () => {
    const preIndex = allocatePopulation('stdp_pre6', 5, { noise: 0 })
    const postIndex = allocatePopulation('stdp_post6', 5, { noise: 0 })
    setNoiseAmplitude(preIndex, 0)
    setNoiseAmplitude(postIndex, 0)

    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'stdp_syn6',
      preIndex,
      postIndex,
      conn.preIndices,
      conn.postIndices,
      { plastic: true }
    )

    // Build eligibility
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(preIndex, 20)
      injectUniformCurrent(postIndex, 20)
      integrate(preIndex, 1.0, false)
      integrate(postIndex, 1.0, false)
      transmit(synIndex)
      updateTraces(synIndex)
      applySTDP(synIndex, false)
    }

    mx.eval(eligibility[synIndex])
    const eligBeforeReset = getScalar(mx.sum(mx.abs(eligibility[synIndex])))

    // Reset
    resetLearning(synIndex)

    mx.eval(eligibility[synIndex])
    const eligAfterReset = getScalar(mx.sum(mx.abs(eligibility[synIndex])))

    releaseSynapseGroup('stdp_syn6')
    releasePopulation('stdp_pre6')
    releasePopulation('stdp_post6')

    return withSlap(
      assertTrue(
        eligBeforeReset > 0.1 && eligAfterReset < 0.001,
        `Before: ${eligBeforeReset.toFixed(4)}, After: ${eligAfterReset.toFixed(
          6
        )}`
      ),
      'Reset clears traces - allows fresh learning episodes'
    )
  })

  endSuite()
}

// Run if executed directly
if (import.meta) {
  runSTDPTests()
}
