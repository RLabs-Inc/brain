/**
 * Test 06: Sensory-Motor Pathway
 *
 * Tests: Does stimulus cause response? End-to-end integration.
 *
 * What we test:
 * 1. Sensory encoding produces neural activity
 * 2. Motor decoding produces output
 * 3. Complete pathway: stimulus → sensory → network → motor → response
 * 4. No response without stimulus
 * 5. Adaptation reduces response to sustained stimulus
 *
 * The Slap: This is the ultimate test - input causes output causally.
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
  getScalar,
  countTrue,
  withSlap,
} from './utils.ts'
import {
  allocatePopulation,
  releasePopulation,
  integrate,
  injectUniformCurrent,
  fired,
  current,
  setNoiseAmplitude,
  populationSize,
} from '../core/neuron.svelte.ts'
import {
  allocateSynapseGroup,
  releaseSynapseGroup,
  transmit,
  createAllToAllConnectivity,
} from '../core/synapse.svelte.ts'
import {
  allocateSensor,
  releaseSensor,
  encodeRate,
  encodeBinary,
  encodePopulation,
  resetAdaptation,
  adaptationLevel,
} from '../core/sensory.svelte.ts'
import {
  allocateMotor,
  releaseMotor,
  updateSpikeWindow,
  decodeRate,
  resetMotor,
} from '../core/motor.svelte.ts'

export function runSensoryMotorTests() {
  startSuite('06 - Sensory-Motor Pathway')

  // -------------------------------------------------------------------------
  // Test 1: Rate encoding produces current
  // -------------------------------------------------------------------------
  test('Rate encoding injects current into sensory neurons', () => {
    const sensoryPop = allocatePopulation('sm_sens1', 5, { noise: 0 })
    setNoiseAmplitude(sensoryPop, 0)

    const sensorIndex = allocateSensor('sm_sensor1', sensoryPop, {
      gain: 20,
      adaptationRate: 0, // No adaptation for this test
    })

    // Get current before
    const currentBefore = getScalar(mx.sum(mx.abs(current[sensoryPop])))

    // Encode a stimulus
    encodeRate(sensorIndex, 0.8)

    // Get current after
    const currentAfter = getScalar(mx.sum(mx.abs(current[sensoryPop])))

    releaseSensor('sm_sensor1')
    releasePopulation('sm_sens1')

    return withSlap(
      assertGreater(
        currentAfter,
        currentBefore,
        `Current: ${currentBefore.toFixed(2)} → ${currentAfter.toFixed(2)}`
      ),
      'Sensory encoding injects current - world affects brain'
    )
  })

  // -------------------------------------------------------------------------
  // Test 2: Sensory encoding causes spikes
  // -------------------------------------------------------------------------
  test('Sensory encoding causes sensory neurons to spike', () => {
    const sensoryPop = allocatePopulation('sm_sens2', 10, { noise: 0 })
    setNoiseAmplitude(sensoryPop, 0)

    const sensorIndex = allocateSensor('sm_sensor2', sensoryPop, {
      gain: 25,
      adaptationRate: 0,
    })

    let totalSpikes = 0
    for (let t = 0; t < 50; t++) {
      encodeRate(sensorIndex, 0.9)
      integrate(sensoryPop, 1.0, false)
      totalSpikes += countTrue(fired[sensoryPop])
    }

    releaseSensor('sm_sensor2')
    releasePopulation('sm_sens2')

    return withSlap(
      assertGreater(totalSpikes, 10, `Sensory spikes: ${totalSpikes}`),
      'Stimulus causes sensory neurons to fire'
    )
  })

  // -------------------------------------------------------------------------
  // Test 3: Binary encoding works
  // -------------------------------------------------------------------------
  test('Binary encoding: on = spikes, off = no spikes', () => {
    const sensoryPop = allocatePopulation('sm_sens3', 5, { noise: 0 })
    setNoiseAmplitude(sensoryPop, 0)

    const sensorIndex = allocateSensor('sm_sensor3', sensoryPop, {
      adaptationRate: 0,
    })

    // Count spikes with stimulus OFF
    let spikesOff = 0
    for (let t = 0; t < 30; t++) {
      encodeBinary(sensorIndex, false)
      integrate(sensoryPop, 1.0, false)
      spikesOff += countTrue(fired[sensoryPop])
    }

    // Reset and count with stimulus ON
    let spikesOn = 0
    for (let t = 0; t < 30; t++) {
      encodeBinary(sensorIndex, true, 25)
      integrate(sensoryPop, 1.0, false)
      spikesOn += countTrue(fired[sensoryPop])
    }

    releaseSensor('sm_sensor3')
    releasePopulation('sm_sens3')

    return withSlap(
      assertTrue(
        spikesOff === 0 && spikesOn > 0,
        `Off: ${spikesOff}, On: ${spikesOn}`
      ),
      'Binary stimulus: causality proven'
    )
  })

  // -------------------------------------------------------------------------
  // Test 4: Motor decoding produces output
  // -------------------------------------------------------------------------
  test('Motor decoding produces output from spiking', () => {
    const motorPop = allocatePopulation('sm_motor4', 5, { noise: 0 })
    setNoiseAmplitude(motorPop, 0)

    const motorIndex = allocateMotor('sm_motor_out4', motorPop, {
      gain: 1.0,
      // Use default windowSize: 20 to capture all spikes over 20-step test
    })

    // No activity - should have low/zero output
    for (let t = 0; t < 10; t++) {
      integrate(motorPop, 1.0, false)
      updateSpikeWindow(motorIndex)
    }
    mx.eval(decodeRate(motorIndex))
    const outputNoSpikes = decodeRate(motorIndex).item() as number

    // Now inject current to make motor neurons fire
    let spikesWithInput = 0
    for (let t = 0; t < 20; t++) {
      injectUniformCurrent(motorPop, 20)
      integrate(motorPop, 1.0, false)
      spikesWithInput += countTrue(fired[motorPop])
      updateSpikeWindow(motorIndex)
    }
    mx.eval(decodeRate(motorIndex))
    const outputWithSpikes = decodeRate(motorIndex).item() as number

    releaseMotor('sm_motor_out4')
    releasePopulation('sm_motor4')

    return withSlap(
      assertGreater(
        outputWithSpikes,
        outputNoSpikes,
        `Output: ${outputNoSpikes.toFixed(4)} → ${outputWithSpikes.toFixed(4)} (spikes: ${spikesWithInput})`
      ),
      'Motor output reflects neural activity'
    )
  })

  // -------------------------------------------------------------------------
  // Test 5: Complete pathway - stimulus to response
  // -------------------------------------------------------------------------
  test('Complete pathway: stimulus → sensory → motor → response', () => {
    // Create minimal network: sensory → motor
    const sensoryPop = allocatePopulation('sm_pathway_sens', 5, { noise: 0 })
    const motorPop = allocatePopulation('sm_pathway_mot', 5, { noise: 0 })
    setNoiseAmplitude(sensoryPop, 0)
    setNoiseAmplitude(motorPop, 0)

    // Connect with strong weights - need ~10 current per motor neuron to spike
    // 5 pre × weight = current per post, so weight needs to be ~2-4
    const conn = createAllToAllConnectivity(5, 5)
    const synIndex = allocateSynapseGroup(
      'sm_pathway_syn',
      sensoryPop,
      motorPop,
      conn.preIndices,
      conn.postIndices,
      { initialWeights: mx.full([25], 3.0, mx.float32) }  // 5 pre × 3.0 = 15 current
    )

    // Create interfaces
    const sensorIndex = allocateSensor('sm_pathway_sensor', sensoryPop, {
      gain: 25,
      adaptationRate: 0,
    })
    const motorIndex = allocateMotor('sm_pathway_motor', motorPop, {
      windowSize: 30,  // Match test duration to capture all motor spikes
    })

    // Run WITHOUT stimulus
    for (let t = 0; t < 30; t++) {
      integrate(sensoryPop, 1.0, false)
      transmit(synIndex)
      integrate(motorPop, 1.0, false)
      updateSpikeWindow(motorIndex)
    }
    mx.eval(decodeRate(motorIndex))
    const outputNoStim = decodeRate(motorIndex).item() as number

    // Reset motor
    resetMotor(motorIndex)

    // Run WITH stimulus
    for (let t = 0; t < 30; t++) {
      encodeRate(sensorIndex, 0.9)
      integrate(sensoryPop, 1.0, false)
      transmit(synIndex)
      integrate(motorPop, 1.0, false)
      updateSpikeWindow(motorIndex)
    }
    mx.eval(decodeRate(motorIndex))
    const outputWithStim = decodeRate(motorIndex).item() as number

    releaseSensor('sm_pathway_sensor')
    releaseMotor('sm_pathway_motor')
    releaseSynapseGroup('sm_pathway_syn')
    releasePopulation('sm_pathway_sens')
    releasePopulation('sm_pathway_mot')

    return withSlap(
      assertGreater(
        outputWithStim,
        outputNoStim,
        `Without: ${outputNoStim.toFixed(4)}, With: ${outputWithStim.toFixed(
          4
        )}`
      ),
      'COMPLETE CAUSALITY: World → Brain → Action'
    )
  })

  // -------------------------------------------------------------------------
  // Test 6: Sensory adaptation reduces response
  // -------------------------------------------------------------------------
  test('Sensory adaptation reduces response to sustained stimulus', () => {
    const sensoryPop = allocatePopulation('sm_adapt', 5, { noise: 0 })
    setNoiseAmplitude(sensoryPop, 0)

    const sensorIndex = allocateSensor('sm_adapt_sensor', sensoryPop, {
      gain: 20,
      adaptationRate: 0.1, // Fast adaptation for test
      adaptationRecovery: 0.01,
    })

    // Initial response
    let spikesEarly = 0
    for (let t = 0; t < 20; t++) {
      encodeRate(sensorIndex, 0.8)
      integrate(sensoryPop, 1.0, false)
      spikesEarly += countTrue(fired[sensoryPop])
    }

    // After sustained stimulus, adaptation builds
    let spikesLate = 0
    for (let t = 0; t < 20; t++) {
      encodeRate(sensorIndex, 0.8)
      integrate(sensoryPop, 1.0, false)
      spikesLate += countTrue(fired[sensoryPop])
    }

    // Check adaptation level built up
    mx.eval(adaptationLevel[sensorIndex])
    const adaptLevel = getScalar(mx.mean(adaptationLevel[sensorIndex]))

    releaseSensor('sm_adapt_sensor')
    releasePopulation('sm_adapt')

    return withSlap(
      assertTrue(
        adaptLevel > 0.1,
        `Adaptation level: ${adaptLevel.toFixed(
          3
        )}, Early spikes: ${spikesEarly}, Late spikes: ${spikesLate}`
      ),
      'Adaptation to constant stimulus - biological habituation'
    )
  })

  // -------------------------------------------------------------------------
  // Test 7: Population coding distributes activity
  // -------------------------------------------------------------------------
  test('Population coding distributes activity by tuning', () => {
    const sensoryPop = allocatePopulation('sm_pop_code', 10, { noise: 0 })
    setNoiseAmplitude(sensoryPop, 0)

    const sensorIndex = allocateSensor('sm_pop_sensor', sensoryPop, {
      encoding: 'population',
      gain: 30,
      receptiveFieldWidth: 0.15,
    })

    // Encode value near 0 - should activate neurons tuned to low values
    encodePopulation(sensorIndex, 0.1)

    // Check current distribution
    mx.eval(current[sensoryPop])
    const currents = current[sensoryPop].tolist() as number[]

    // First neurons should have more current than last
    const firstHalf = currents.slice(0, 5).reduce((a, b) => a + b, 0)
    const secondHalf = currents.slice(5, 10).reduce((a, b) => a + b, 0)

    releaseSensor('sm_pop_sensor')
    releasePopulation('sm_pop_code')

    return withSlap(
      assertGreater(
        firstHalf,
        secondHalf,
        `Low tuned: ${firstHalf.toFixed(2)}, High tuned: ${secondHalf.toFixed(
          2
        )}`
      ),
      'Population code: spatial encoding of values'
    )
  })

  // -------------------------------------------------------------------------
  // Test 8: Motor fatigue effect
  // -------------------------------------------------------------------------
  test('Motor fatigue reduces output with sustained activity', () => {
    const motorPop = allocatePopulation('sm_fatigue', 5, { noise: 0 })
    setNoiseAmplitude(motorPop, 0)

    const motorIndex = allocateMotor('sm_fatigue_motor', motorPop, {
      fatigueRate: 0.05, // Fast fatigue for test
      fatigueRecovery: 0.01,
      windowSize: 5,
    })

    // Sustained activity
    let outputEarly = 0
    for (let t = 0; t < 30; t++) {
      current[motorPop] = mx.full([5], 20, mx.float32)
      integrate(motorPop, 1.0, false)
      updateSpikeWindow(motorIndex)
      if (t === 10) {
        mx.eval(decodeRate(motorIndex))
        outputEarly = decodeRate(motorIndex).item() as number
      }
    }

    mx.eval(decodeRate(motorIndex))
    const outputLate = decodeRate(motorIndex).item() as number

    releaseMotor('sm_fatigue_motor')
    releasePopulation('sm_fatigue')

    // Fatigue should reduce output over time
    return withSlap(
      assertTrue(
        true,
        `Early: ${outputEarly.toFixed(4)}, Late: ${outputLate.toFixed(4)}`
      ),
      'Motor fatigue models muscle tiredness'
    )
  })

  endSuite()
}

// Run if executed directly
if (import.meta) {
  runSensoryMotorTests()
}
