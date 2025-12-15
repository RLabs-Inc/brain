/**
 * Test 05: Genome System
 *
 * Tests: Does a genome specification create a valid network?
 *
 * What we test:
 * 1. Valid genome passes validation
 * 2. Invalid genome fails validation
 * 3. Genome loading creates populations
 * 4. Genome loading creates synapses
 * 5. Neuron types are set correctly
 * 6. Reflex pathways are created
 *
 * The Slap: Genome must create EXACTLY what we specified.
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
  assertFalse,
  assertGreater,
  getScalar,
  getArray,
  withSlap,
} from './utils.ts'
import type { Genome } from '../core/genome.svelte.ts'
import {
  validateGenome,
  loadGenome,
  getNetworkForGenome,
  getPopulationForNeuron,
  getPopulationsForGenome,
  getSynapseGroupsForGenome,
  isGenomeLoaded,
} from '../core/genome.svelte.ts'
import { populationSize, isExcitatory } from '../core/neuron.svelte.ts'
import {
  networkPopulations,
  networkSynapseGroups,
} from '../core/network.svelte.ts'
import { weights } from '../core/synapse.svelte.ts'

// Test genomes
const validGenome: Genome = {
  name: 'test_valid',
  neurons: [
    { id: 'input', size: 5, type: 'RS', role: 'sensory' },
    { id: 'hidden', size: 10, type: 'RS', role: 'inter' },
    { id: 'output', size: 3, type: 'RS', role: 'motor' },
  ],
  synapses: [
    { pre: 'input', post: 'hidden', pattern: 'all-to-all', plastic: true },
    { pre: 'hidden', post: 'output', pattern: 'all-to-all', plastic: true },
  ],
}

const invalidGenome: Genome = {
  name: '', // Invalid: empty name
  neurons: [], // Invalid: no neurons
  synapses: [],
}

const genomeWithReflex: Genome = {
  name: 'test_reflex',
  neurons: [
    { id: 'sensor', size: 2, type: 'RS', role: 'sensory' },
    { id: 'motor', size: 2, type: 'RS', role: 'motor' },
  ],
  synapses: [],
  reflexes: [
    {
      name: 'quick_response',
      pathway: ['sensor', 'motor'],
      strength: 0.4,
      plastic: false,
    },
  ],
}

const genomeWithTypes: Genome = {
  name: 'test_types',
  neurons: [
    { id: 'exc', size: 5, type: 'RS' }, // Excitatory
    { id: 'inh', size: 3, type: 'FS' }, // Inhibitory
  ],
  synapses: [
    { pre: 'exc', post: 'inh', pattern: 'all-to-all', plastic: false },
    { pre: 'inh', post: 'exc', pattern: 'all-to-all', plastic: false },
  ],
}

export function runGenomeTests() {
  startSuite('05 - Genome System')

  // -------------------------------------------------------------------------
  // Test 1: Valid genome passes validation
  // -------------------------------------------------------------------------
  test('Valid genome passes validation', () => {
    const result = validateGenome(validGenome)

    return withSlap(
      assertTrue(result.valid, `Errors: ${result.errors.join(', ')}`),
      'Validation catches structural issues'
    )
  })

  // -------------------------------------------------------------------------
  // Test 2: Invalid genome fails validation
  // -------------------------------------------------------------------------
  test('Invalid genome fails validation', () => {
    const result = validateGenome(invalidGenome)

    return withSlap(
      assertFalse(
        result.valid,
        `Expected invalid, got valid. Errors: ${result.errors.join(', ')}`
      ),
      'Validation rejects bad genomes - safety check'
    )
  })

  // -------------------------------------------------------------------------
  // Test 3: Missing neuron reference caught
  // -------------------------------------------------------------------------
  test('Missing neuron reference in synapse caught', () => {
    const badGenome: Genome = {
      name: 'bad_ref',
      neurons: [{ id: 'A', size: 1 }],
      synapses: [{ pre: 'A', post: 'B', pattern: 'all-to-all' }], // B doesn't exist
    }

    const result = validateGenome(badGenome)

    return withSlap(
      assertFalse(result.valid, `Should fail: ${result.errors.join(', ')}`),
      'Catches dangling references - prevents runtime errors'
    )
  })

  // -------------------------------------------------------------------------
  // Test 4: Genome loading creates network
  // -------------------------------------------------------------------------
  test('Genome loading creates network', () => {
    const networkIndex = loadGenome(validGenome)
    const loaded = isGenomeLoaded('test_valid')

    return withSlap(
      assertTrue(
        loaded && networkIndex >= 0,
        `Network index: ${networkIndex}, Loaded: ${loaded}`
      ),
      'Network created from genome spec'
    )
  })

  // -------------------------------------------------------------------------
  // Test 5: Correct number of populations created
  // -------------------------------------------------------------------------
  test('Correct number of populations created', () => {
    const pops = getPopulationsForGenome('test_valid')
    const count = pops?.size ?? 0

    return withSlap(
      assertEqual(count, 3, `Populations: ${count}`),
      'Each neuron spec becomes a population'
    )
  })

  // -------------------------------------------------------------------------
  // Test 6: Population sizes match spec
  // -------------------------------------------------------------------------
  test('Population sizes match genome spec', () => {
    const inputPop = getPopulationForNeuron('test_valid', 'input')
    const hiddenPop = getPopulationForNeuron('test_valid', 'hidden')
    const outputPop = getPopulationForNeuron('test_valid', 'output')

    const inputSize = inputPop !== undefined ? populationSize[inputPop] : -1
    const hiddenSize = hiddenPop !== undefined ? populationSize[hiddenPop] : -1
    const outputSize = outputPop !== undefined ? populationSize[outputPop] : -1

    const correct = inputSize === 5 && hiddenSize === 10 && outputSize === 3

    return withSlap(
      assertTrue(
        correct,
        `Sizes: input=${inputSize}, hidden=${hiddenSize}, output=${outputSize}`
      ),
      'Sizes exactly as specified in genome'
    )
  })

  // -------------------------------------------------------------------------
  // Test 7: Correct number of synapse groups created
  // -------------------------------------------------------------------------
  test('Correct number of synapse groups created', () => {
    const syns = getSynapseGroupsForGenome('test_valid')
    const count = syns?.size ?? 0

    return withSlap(
      assertEqual(count, 2, `Synapse groups: ${count}`),
      'Each synapse spec becomes a group'
    )
  })

  // -------------------------------------------------------------------------
  // Test 8: Neuron types set correctly
  // -------------------------------------------------------------------------
  test('Neuron types (excitatory/inhibitory) set correctly', () => {
    loadGenome(genomeWithTypes)

    const excPop = getPopulationForNeuron('test_types', 'exc')
    const inhPop = getPopulationForNeuron('test_types', 'inh')

    const excIsExc = excPop !== undefined ? isExcitatory[excPop] : false
    const inhIsExc = inhPop !== undefined ? isExcitatory[inhPop] : false

    return withSlap(
      assertTrue(
        excIsExc && !inhIsExc,
        `Exc pop excitatory: ${excIsExc}, Inh pop excitatory: ${inhIsExc}`
      ),
      "Dale's Law respected from genome spec"
    )
  })

  // -------------------------------------------------------------------------
  // Test 9: Reflex pathway creates synapses
  // -------------------------------------------------------------------------
  test('Reflex pathway creates direct synapses', () => {
    loadGenome(genomeWithReflex)

    const syns = getSynapseGroupsForGenome('test_reflex')
    const count = syns?.size ?? 0

    // Should have 1 synapse from the reflex (sensor → motor)
    return withSlap(
      assertGreater(count, 0, `Reflex synapse groups: ${count}`),
      'Reflex pathways are wired as specified'
    )
  })

  // -------------------------------------------------------------------------
  // Test 10: Reflex has correct initial weight
  // -------------------------------------------------------------------------
  test('Reflex synapses have specified strength', () => {
    const syns = getSynapseGroupsForGenome('test_reflex')

    let foundWeight = false
    let actualWeight = 0

    if (syns) {
      for (const [, synIndex] of syns) {
        mx.eval(weights[synIndex])
        const w = getArray(weights[synIndex])
        if (w.length > 0) {
          actualWeight = w[0]
          foundWeight = Math.abs(actualWeight - 0.4) < 0.01
          if (foundWeight) break
        }
      }
    }

    return withSlap(
      assertTrue(
        foundWeight,
        `Reflex weight: ${actualWeight.toFixed(3)} (expected 0.4)`
      ),
      'Innate connection strength as specified in genome'
    )
  })

  // -------------------------------------------------------------------------
  // Test 11: One-to-one pattern validation
  // -------------------------------------------------------------------------
  test('One-to-one pattern requires equal sizes', () => {
    const badGenome: Genome = {
      name: 'bad_one_to_one',
      neurons: [
        { id: 'A', size: 5 },
        { id: 'B', size: 3 }, // Different size!
      ],
      synapses: [{ pre: 'A', post: 'B', pattern: 'one-to-one' }],
    }

    const result = validateGenome(badGenome)

    return withSlap(
      assertFalse(result.valid, `Should fail: ${result.errors.join(', ')}`),
      'Catches impossible connectivity patterns'
    )
  })

  // -------------------------------------------------------------------------
  // Test 12: Duplicate neuron ID caught
  // -------------------------------------------------------------------------
  test('Duplicate neuron IDs caught', () => {
    const badGenome: Genome = {
      name: 'dup_ids',
      neurons: [
        { id: 'same', size: 5 },
        { id: 'same', size: 3 }, // Duplicate!
      ],
      synapses: [],
    }

    const result = validateGenome(badGenome)

    return withSlap(
      assertFalse(result.valid, `Should fail: ${result.errors.join(', ')}`),
      'Catches duplicate IDs - prevents confusion'
    )
  })

  endSuite()
}

// Run if executed directly
if (import.meta) {
  runGenomeTests()
}
